import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard127

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound24772
def owner : Owner := ⟨.program ⟨257⟩, ⟨21288⟩⟩
def transferEvent : Nat := 24772
def frameStart : Nat := 24739
def rule : BoundRule := .identity (.predecessor 0 24771 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24771 .coefficient)
      LeftBound24768.bound (LeftBound24768.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24770RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24768.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24768.derived selector witness)

def rawBound : CoeffClass := LeftBound24768.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24768.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound24768.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound24772

namespace LeftBound24789
def owner : Owner := ⟨.program ⟨257⟩, ⟨23170⟩⟩
def transferEvent : Nat := 24789
def frameStart : Nat := 24739
def rule : BoundRule := .sum [.predecessor 0 24787 .coefficient, .predecessor 1 24788 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24787 .coefficient)
      LeftBound24772.bound (LeftBound24772.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound24772.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24788 .coefficient)
      LeftAuthority24785.bound (LeftAuthority24785.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority24785.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24772.bound, LeftAuthority24785.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24772.bound, LeftAuthority24785.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound24772.actual selector witness, LeftAuthority24785.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24789

namespace LeftBound24792
def owner : Owner := ⟨.program ⟨257⟩, ⟨23171⟩⟩
def transferEvent : Nat := 24792
def frameStart : Nat := 24739
def rule : BoundRule := .identity (.predecessor 0 24791 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24791 .coefficient)
      LeftBound24789.bound (LeftBound24789.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound24789.derived selector witness)

def rawBound : CoeffClass := LeftBound24789.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24789.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound24789.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound24792

namespace LeftBound24798
def owner : Owner := ⟨.program ⟨257⟩, ⟨23172⟩⟩
def transferEvent : Nat := 24798
def frameStart : Nat := 24739
def rule : BoundRule := .product (.predecessor 0 24796 .coefficient) (.predecessor 1 24797 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24796 .coefficient)
      LeftAuthority24794.bound (LeftAuthority24794.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24795RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24794.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24794.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24797 .coefficient)
      LeftBound24792.bound (LeftBound24792.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24793RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24792.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24792.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority24794.bound LeftBound24792.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24794.bound, LeftBound24792.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority24794.actual selector witness) * (LeftBound24792.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24798

namespace LeftBound24814
def owner : Owner := ⟨.program ⟨257⟩, ⟨9575⟩⟩
def transferEvent : Nat := 24814
def frameStart : Nat := 24739
def rule : BoundRule := .scale (.predecessor 0 24812 .coefficient) (.value (.predecessor 1 24813 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24812 .coefficient)
      LeftAuthority24810.bound (LeftAuthority24810.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24811RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24810.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24810.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24813 .coefficient)
      LeftAuthority24801.bound (LeftAuthority24801.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority24801.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority24810.bound LeftAuthority24801.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24810.bound, LeftAuthority24801.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority24810.actual selector witness) * (LeftAuthority24801.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound24814

namespace LeftBound24817
def owner : Owner := ⟨.program ⟨257⟩, ⟨7286⟩⟩
def transferEvent : Nat := 24817
def frameStart : Nat := 24739
def rule : BoundRule := .identity (.predecessor 0 24816 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24816 .coefficient)
      LeftAuthority24804.bound (LeftAuthority24804.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24805RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24804.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24804.derived selector witness)

def rawBound : CoeffClass := LeftAuthority24804.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24804.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority24804.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound24817

namespace LeftBound24821
def owner : Owner := ⟨.program ⟨257⟩, ⟨9576⟩⟩
def transferEvent : Nat := 24821
def frameStart : Nat := 24739
def rule : BoundRule := .product (.predecessor 0 24819 .coefficient) (.predecessor 1 24820 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24819 .coefficient)
      LeftBound24817.bound (LeftBound24817.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24818RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24817.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24817.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24820 .coefficient)
      LeftBound24814.bound (LeftBound24814.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24814.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24814.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound24817.bound LeftBound24814.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24817.bound, LeftBound24814.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound24817.actual selector witness) * (LeftBound24814.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24821

namespace LeftBound24826
def owner : Owner := ⟨.program ⟨257⟩, ⟨23173⟩⟩
def transferEvent : Nat := 24826
def frameStart : Nat := 24739
def rule : BoundRule := .sum [.predecessor 0 24824 .coefficient, .predecessor 1 24825 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24824 .coefficient)
      LeftBound24821.bound (LeftBound24821.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24823RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24821.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24821.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24825 .coefficient)
      LeftBound24798.bound (LeftBound24798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24800RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24798.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24798.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24821.bound, LeftBound24798.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24821.bound, LeftBound24798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound24821.actual selector witness, LeftBound24798.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24826

namespace LeftBound24830
def owner : Owner := ⟨.program ⟨257⟩, ⟨23346⟩⟩
def transferEvent : Nat := 24830
def frameStart : Nat := 24739
def rule : BoundRule := .product (.predecessor 0 24828 .coefficient) (.predecessor 1 24829 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24828 .coefficient)
      LeftBound24826.bound (LeftBound24826.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24827RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24826.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24826.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24829 .coefficient)
      LeftAuthority24783.bound (LeftAuthority24783.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24784RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24783.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24783.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound24826.bound LeftAuthority24783.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24826.bound, LeftAuthority24783.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound24826.actual selector witness) * (LeftAuthority24783.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24830

namespace LeftBound24841
def owner : Owner := ⟨.program ⟨257⟩, ⟨21740⟩⟩
def transferEvent : Nat := 24841
def frameStart : Nat := 24739
def rule : BoundRule := .product (.predecessor 0 24839 .coefficient) (.predecessor 1 24840 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24839 .coefficient)
      LeftAuthority24794.bound (LeftAuthority24794.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24795RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24794.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24794.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24840 .coefficient)
      LeftAuthority24837.bound (LeftAuthority24837.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact24838RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24837.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24837.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority24794.bound LeftAuthority24837.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24794.bound, LeftAuthority24837.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority24794.actual selector witness) * (LeftAuthority24837.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24841

namespace LeftBound24849
def owner : Owner := ⟨.program ⟨257⟩, ⟨21741⟩⟩
def transferEvent : Nat := 24849
def frameStart : Nat := 24739
def rule : BoundRule := .sum [.predecessor 0 24847 .coefficient, .predecessor 1 24848 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24847 .coefficient)
      LeftAuthority24845.bound (LeftAuthority24845.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact24846RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24845.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24845.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24848 .coefficient)
      LeftBound24841.bound (LeftBound24841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact24843RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24841.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24841.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority24845.bound, LeftBound24841.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24845.bound, LeftBound24841.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority24845.actual selector witness, LeftBound24841.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24849

namespace LeftBound24853
def owner : Owner := ⟨.program ⟨257⟩, ⟨23347⟩⟩
def transferEvent : Nat := 24853
def frameStart : Nat := 24739
def rule : BoundRule := .sum [.predecessor 0 24851 .coefficient, .predecessor 1 24852 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24851 .coefficient)
      LeftBound24849.bound (LeftBound24849.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact24850RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24849.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24849.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24852 .coefficient)
      LeftBound24830.bound (LeftBound24830.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact24835RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24830.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24830.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24849.bound, LeftBound24830.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24849.bound, LeftBound24830.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound24849.actual selector witness, LeftBound24830.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24853

namespace LeftBound24866
def owner : Owner := ⟨.program ⟨257⟩, ⟨23345⟩⟩
def transferEvent : Nat := 24866
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 24864 .coefficient, .predecessor 1 24865 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24864 .coefficient)
      LeftBound24687.bound (LeftBound24687.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact24863RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24687.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24687.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24865 .coefficient)
      LeftBound24670.bound (LeftBound24670.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24670.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24670.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24687.bound, LeftBound24670.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24687.bound, LeftBound24670.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound24687.actual selector witness, LeftBound24670.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24866

namespace LeftBound24869
def owner : Owner := ⟨.program ⟨257⟩, ⟨23345⟩⟩
def transferEvent : Nat := 24869
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 24863 .summary, .result 24677 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 24863 .summary)
      LeftBound24689.bound (LeftBound24689.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨22285⟩⟩) (rawTerms := some (Proof.Events097.exact24863RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24689.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 24677 .summary)
      LeftBound24672.bound (LeftBound24672.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23344⟩⟩) (rawTerms := some (Proof.Events096.exact24677RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24672.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24689.bound, LeftBound24672.bound]
def bound : CoeffClass := .finite ⟨2997834576566628384768, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24689.bound, LeftBound24672.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound24689.actual selector witness, LeftBound24672.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24869

namespace LeftBound24873
def owner : Owner := ⟨.program ⟨257⟩, ⟨23604⟩⟩
def transferEvent : Nat := 24873
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 24871 .coefficient) (.predecessor 1 24872 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24871 .coefficient)
      LeftBound24866.bound (LeftBound24866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact24870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24866.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24866.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24872 .coefficient)
      LeftAuthority24573.bound (LeftAuthority24573.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24574RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24573.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24573.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound24866.bound LeftAuthority24573.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24866.bound, LeftAuthority24573.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound24866.actual selector witness) * (LeftAuthority24573.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24873

namespace LeftBound24874
def owner : Owner := ⟨.program ⟨257⟩, ⟨23604⟩⟩
def transferEvent : Nat := 24874
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨23602⟩⟩]⟩ [⟨.result 24574 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 24574 .coefficient)
      LeftAuthority24573.bound (LeftAuthority24573.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨23602⟩⟩) (rawTerms := some (Proof.Events095.exact24574RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24573.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24573.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority24573.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24573.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority24573.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound24874

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
