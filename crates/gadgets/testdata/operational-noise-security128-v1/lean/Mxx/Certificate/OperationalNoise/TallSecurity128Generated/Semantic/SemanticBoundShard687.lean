import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard682
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard686

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound105869
def owner : Owner := ⟨.program ⟨257⟩, ⟨46753⟩⟩
def transferEvent : Nat := 105869
def frameStart : Nat := 105782
def rule : BoundRule := .sum [.predecessor 0 105867 .coefficient, .predecessor 1 105868 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105867 .coefficient)
      LeftBound105864.bound (LeftBound105864.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105866RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105864.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105864.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 105868 .coefficient)
      LeftBound105841.bound (LeftBound105841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105843RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105841.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105841.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105864.bound, LeftBound105841.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105864.bound, LeftBound105841.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound105864.actual selector witness, LeftBound105841.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105869

namespace LeftBound105873
def owner : Owner := ⟨.program ⟨257⟩, ⟨46993⟩⟩
def transferEvent : Nat := 105873
def frameStart : Nat := 105782
def rule : BoundRule := .product (.predecessor 0 105871 .coefficient) (.predecessor 1 105872 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105871 .coefficient)
      LeftBound105869.bound (LeftBound105869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105869.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105869.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 105872 .coefficient)
      LeftAuthority105826.bound (LeftAuthority105826.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105827RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105826.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105826.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound105869.bound LeftAuthority105826.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105869.bound, LeftAuthority105826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound105869.actual selector witness) * (LeftAuthority105826.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105873

namespace LeftBound105884
def owner : Owner := ⟨.program ⟨257⟩, ⟨45478⟩⟩
def transferEvent : Nat := 105884
def frameStart : Nat := 105782
def rule : BoundRule := .product (.predecessor 0 105882 .coefficient) (.predecessor 1 105883 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105882 .coefficient)
      LeftAuthority105837.bound (LeftAuthority105837.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105838RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105837.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105837.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 105883 .coefficient)
      LeftAuthority105880.bound (LeftAuthority105880.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105881RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105880.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105880.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority105837.bound LeftAuthority105880.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105837.bound, LeftAuthority105880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority105837.actual selector witness) * (LeftAuthority105880.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105884

namespace LeftBound105892
def owner : Owner := ⟨.program ⟨257⟩, ⟨45479⟩⟩
def transferEvent : Nat := 105892
def frameStart : Nat := 105782
def rule : BoundRule := .sum [.predecessor 0 105890 .coefficient, .predecessor 1 105891 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105890 .coefficient)
      LeftAuthority105888.bound (LeftAuthority105888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105889RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105888.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105888.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 105891 .coefficient)
      LeftBound105884.bound (LeftBound105884.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105886RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105884.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105884.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority105888.bound, LeftBound105884.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105888.bound, LeftBound105884.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority105888.actual selector witness, LeftBound105884.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105892

namespace LeftBound105896
def owner : Owner := ⟨.program ⟨257⟩, ⟨46994⟩⟩
def transferEvent : Nat := 105896
def frameStart : Nat := 105782
def rule : BoundRule := .sum [.predecessor 0 105894 .coefficient, .predecessor 1 105895 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105894 .coefficient)
      LeftBound105892.bound (LeftBound105892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105892.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105892.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 105895 .coefficient)
      LeftBound105873.bound (LeftBound105873.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105873.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105873.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105892.bound, LeftBound105873.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105892.bound, LeftBound105873.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound105892.actual selector witness, LeftBound105873.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105896

namespace LeftBound105909
def owner : Owner := ⟨.program ⟨257⟩, ⟨46992⟩⟩
def transferEvent : Nat := 105909
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 105907 .coefficient, .predecessor 1 105908 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105907 .coefficient)
      LeftBound105730.bound (LeftBound105730.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105906RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105730.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105730.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 105908 .coefficient)
      LeftBound105713.bound (LeftBound105713.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105720RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105713.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105713.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105730.bound, LeftBound105713.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105730.bound, LeftBound105713.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound105730.actual selector witness, LeftBound105713.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105909

namespace LeftBound105912
def owner : Owner := ⟨.program ⟨257⟩, ⟨46992⟩⟩
def transferEvent : Nat := 105912
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 105906 .summary, .result 105720 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 105906 .summary)
      LeftBound105732.bound (LeftBound105732.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨45922⟩⟩) (rawTerms := some (Proof.Events413.exact105906RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105732.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 105720 .summary)
      LeftBound105715.bound (LeftBound105715.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨46991⟩⟩) (rawTerms := some (Proof.Events412.exact105720RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105715.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105732.bound, LeftBound105715.bound]
def bound : CoeffClass := .finite ⟨2998328565150755586048, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105732.bound, LeftBound105715.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound105732.actual selector witness, LeftBound105715.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105912

namespace LeftBound105916
def owner : Owner := ⟨.program ⟨257⟩, ⟨47376⟩⟩
def transferEvent : Nat := 105916
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 105914 .coefficient) (.predecessor 1 105915 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105914 .coefficient)
      LeftBound105909.bound (LeftBound105909.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105913RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105909.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105909.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 105915 .coefficient)
      LeftAuthority105635.bound (LeftAuthority105635.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105635.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105635.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound105909.bound LeftAuthority105635.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105909.bound, LeftAuthority105635.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound105909.actual selector witness) * (LeftAuthority105635.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105916

namespace LeftBound105917
def owner : Owner := ⟨.program ⟨257⟩, ⟨47376⟩⟩
def transferEvent : Nat := 105917
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨47374⟩⟩]⟩ [⟨.result 105636 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 105636 .coefficient)
      LeftAuthority105635.bound (LeftAuthority105635.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨47374⟩⟩) (rawTerms := some (Proof.Events412.exact105636RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105635.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105635.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority105635.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105635.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority105635.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound105917

namespace LeftBound105918
def owner : Owner := ⟨.program ⟨257⟩, ⟨47376⟩⟩
def transferEvent : Nat := 105918
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 105913 .summary) (.transfer 105917) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 105913 .summary)
      LeftBound105912.bound (LeftBound105912.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨46992⟩⟩) (rawTerms := some (Proof.Events413.exact105913RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105912.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 105917)
      LeftBound105917.bound (LeftBound105917.actual selector witness) := by
  exact .transfer (LeftBound105917.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound105912.bound LeftBound105917.bound
def bound : CoeffClass := .finite ⟨32194307824962751379413684715520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105912.bound, LeftBound105917.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound105912.actual selector witness) * (LeftBound105917.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105918

namespace LeftBound105929
def owner : Owner := ⟨.program ⟨257⟩, ⟨46238⟩⟩
def transferEvent : Nat := 105929
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 105927 .coefficient) (.value (.predecessor 1 105928 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105927 .coefficient)
      LeftAuthority105925.bound (LeftAuthority105925.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105926RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105925.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105925.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 105928 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority105925.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105925.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority105925.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound105929

namespace LeftBound105933
def owner : Owner := ⟨.program ⟨257⟩, ⟨46239⟩⟩
def transferEvent : Nat := 105933
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 105931 .coefficient) (.predecessor 1 105932 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 105931 .coefficient)
      LeftBound105242.bound (LeftBound105242.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105242.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 105932 .coefficient)
      LeftBound105929.bound (LeftBound105929.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105929.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105929.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound105242.bound LeftBound105929.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105242.bound, LeftBound105929.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound105242.actual selector witness) * (LeftBound105929.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105933

namespace LeftBound105934
def owner : Owner := ⟨.program ⟨257⟩, ⟨46239⟩⟩
def transferEvent : Nat := 105934
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨46236⟩⟩]⟩ [⟨.result 105926 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 105926 .coefficient)
      LeftAuthority105925.bound (LeftAuthority105925.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨46236⟩⟩) (rawTerms := some (Proof.Events413.exact105926RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105925.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105925.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority105925.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105925.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority105925.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound105934

namespace LeftBound105935
def owner : Owner := ⟨.program ⟨257⟩, ⟨46239⟩⟩
def transferEvent : Nat := 105935
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 105245 .summary) (.transfer 105934) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 105245 .summary)
      LeftBound105243.bound (LeftBound105243.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5770⟩⟩) (rawTerms := some (Proof.Events411.exact105245RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 105934)
      LeftBound105934.bound (LeftBound105934.actual selector witness) := by
  exact .transfer (LeftBound105934.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound105243.bound LeftBound105934.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105243.bound, LeftBound105934.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound105243.actual selector witness) * (LeftBound105934.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105935

namespace LeftBound106030
def owner : Owner := ⟨.program ⟨257⟩, ⟨45477⟩⟩
def transferEvent : Nat := 106030
def frameStart : Nat := 105991
def rule : BoundRule := .identity (.predecessor 0 106029 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 106029 .coefficient)
      LeftAuthority106027.bound (LeftAuthority106027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106027.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106027.derived selector witness)

def rawBound : CoeffClass := LeftAuthority106027.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority106027.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound106030

namespace LeftBound106047
def owner : Owner := ⟨.program ⟨257⟩, ⟨46830⟩⟩
def transferEvent : Nat := 106047
def frameStart : Nat := 105991
def rule : BoundRule := .sum [.predecessor 0 106045 .coefficient, .predecessor 1 106046 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 106045 .coefficient)
      LeftBound106030.bound (LeftBound106030.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound106030.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 106046 .coefficient)
      LeftAuthority106043.bound (LeftAuthority106043.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority106043.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106030.bound, LeftAuthority106043.bound]
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106030.bound, LeftAuthority106043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound106030.actual selector witness, LeftAuthority106043.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106047

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
