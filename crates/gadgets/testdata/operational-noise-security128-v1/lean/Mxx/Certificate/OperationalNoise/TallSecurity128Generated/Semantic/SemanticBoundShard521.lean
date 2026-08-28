import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard478
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard479
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard520

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound81499
def owner : Owner := ⟨.program ⟨257⟩, ⟨57838⟩⟩
def transferEvent : Nat := 81499
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 81497 .coefficient) (.value (.predecessor 1 81498 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 81497 .coefficient)
      LeftAuthority81495.bound (LeftAuthority81495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81496RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81495.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81495.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 81498 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority81495.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81495.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority81495.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound81499

namespace LeftBound81503
def owner : Owner := ⟨.program ⟨257⟩, ⟨57839⟩⟩
def transferEvent : Nat := 81503
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 81501 .coefficient) (.predecessor 1 81502 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 81501 .coefficient)
      LeftBound75992.bound (LeftBound75992.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75992.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 81502 .coefficient)
      LeftBound81499.bound (LeftBound81499.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81499.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81499.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound75992.bound LeftBound81499.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75992.bound, LeftBound81499.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound75992.actual selector witness) * (LeftBound81499.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81503

namespace LeftBound81504
def owner : Owner := ⟨.program ⟨257⟩, ⟨57839⟩⟩
def transferEvent : Nat := 81504
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨57836⟩⟩]⟩ [⟨.result 81496 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 81496 .coefficient)
      LeftAuthority81495.bound (LeftAuthority81495.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨57836⟩⟩) (rawTerms := some (Proof.Events318.exact81496RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81495.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81495.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority81495.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81495.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority81495.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound81504

namespace LeftBound81505
def owner : Owner := ⟨.program ⟨257⟩, ⟨57839⟩⟩
def transferEvent : Nat := 81505
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 75995 .summary) (.transfer 81504) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75995 .summary)
      LeftBound75993.bound (LeftBound75993.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨10368⟩⟩) (rawTerms := some (Proof.Events296.exact75995RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 81504)
      LeftBound81504.bound (LeftBound81504.actual selector witness) := by
  exact .transfer (LeftBound81504.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound75993.bound LeftBound81504.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75993.bound, LeftBound81504.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound75993.actual selector witness) * (LeftBound81504.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81505

namespace LeftBound81600
def owner : Owner := ⟨.program ⟨257⟩, ⟨56897⟩⟩
def transferEvent : Nat := 81600
def frameStart : Nat := 81561
def rule : BoundRule := .identity (.predecessor 0 81599 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 81599 .coefficient)
      LeftAuthority81597.bound (LeftAuthority81597.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81598RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81597.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81597.derived selector witness)

def rawBound : CoeffClass := LeftAuthority81597.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81597.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority81597.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound81600

namespace LeftBound81617
def owner : Owner := ⟨.program ⟨257⟩, ⟨58350⟩⟩
def transferEvent : Nat := 81617
def frameStart : Nat := 81561
def rule : BoundRule := .sum [.predecessor 0 81615 .coefficient, .predecessor 1 81616 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 81615 .coefficient)
      LeftBound81600.bound (LeftBound81600.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound81600.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 81616 .coefficient)
      LeftAuthority81613.bound (LeftAuthority81613.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority81613.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81600.bound, LeftAuthority81613.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81600.bound, LeftAuthority81613.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound81600.actual selector witness, LeftAuthority81613.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81617

namespace LeftBound81620
def owner : Owner := ⟨.program ⟨257⟩, ⟨58351⟩⟩
def transferEvent : Nat := 81620
def frameStart : Nat := 81561
def rule : BoundRule := .identity (.predecessor 0 81619 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 81619 .coefficient)
      LeftBound81617.bound (LeftBound81617.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound81617.derived selector witness)

def rawBound : CoeffClass := LeftBound81617.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81617.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound81617.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound81620

namespace LeftBound81626
def owner : Owner := ⟨.program ⟨257⟩, ⟨58352⟩⟩
def transferEvent : Nat := 81626
def frameStart : Nat := 81561
def rule : BoundRule := .product (.predecessor 0 81624 .coefficient) (.predecessor 1 81625 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 81624 .coefficient)
      LeftAuthority81622.bound (LeftAuthority81622.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81623RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81622.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81622.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 81625 .coefficient)
      LeftBound81620.bound (LeftBound81620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81621RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81620.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81620.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority81622.bound LeftBound81620.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81622.bound, LeftBound81620.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority81622.actual selector witness) * (LeftBound81620.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81626

namespace LeftBound81634
def owner : Owner := ⟨.program ⟨257⟩, ⟨58353⟩⟩
def transferEvent : Nat := 81634
def frameStart : Nat := 81561
def rule : BoundRule := .sum [.predecessor 0 81632 .coefficient, .predecessor 1 81633 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 81632 .coefficient)
      LeftAuthority81630.bound (LeftAuthority81630.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81631RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81630.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81630.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 81633 .coefficient)
      LeftBound81626.bound (LeftBound81626.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81626.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81626.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority81630.bound, LeftBound81626.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81630.bound, LeftBound81626.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority81630.actual selector witness, LeftBound81626.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81634

namespace LeftBound81638
def owner : Owner := ⟨.program ⟨257⟩, ⟨59099⟩⟩
def transferEvent : Nat := 81638
def frameStart : Nat := 81561
def rule : BoundRule := .product (.predecessor 0 81636 .coefficient) (.predecessor 1 81637 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 81636 .coefficient)
      LeftBound81634.bound (LeftBound81634.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81634.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81634.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 81637 .coefficient)
      LeftAuthority81611.bound (LeftAuthority81611.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81612RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81611.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81611.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound81634.bound LeftAuthority81611.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81634.bound, LeftAuthority81611.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound81634.actual selector witness) * (LeftAuthority81611.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81638

namespace LeftBound81649
def owner : Owner := ⟨.program ⟨257⟩, ⟨57237⟩⟩
def transferEvent : Nat := 81649
def frameStart : Nat := 81561
def rule : BoundRule := .product (.predecessor 0 81647 .coefficient) (.predecessor 1 81648 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 81647 .coefficient)
      LeftAuthority81622.bound (LeftAuthority81622.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81623RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81622.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81622.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 81648 .coefficient)
      LeftAuthority81645.bound (LeftAuthority81645.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81646RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81645.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81645.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority81622.bound LeftAuthority81645.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81622.bound, LeftAuthority81645.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority81622.actual selector witness) * (LeftAuthority81645.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81649

namespace LeftBound81657
def owner : Owner := ⟨.program ⟨257⟩, ⟨57238⟩⟩
def transferEvent : Nat := 81657
def frameStart : Nat := 81561
def rule : BoundRule := .sum [.predecessor 0 81655 .coefficient, .predecessor 1 81656 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 81655 .coefficient)
      LeftAuthority81653.bound (LeftAuthority81653.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81654RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81653.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81653.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 81656 .coefficient)
      LeftBound81649.bound (LeftBound81649.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81651RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81649.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81649.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority81653.bound, LeftBound81649.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81653.bound, LeftBound81649.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority81653.actual selector witness, LeftBound81649.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81657

namespace LeftBound81661
def owner : Owner := ⟨.program ⟨257⟩, ⟨59103⟩⟩
def transferEvent : Nat := 81661
def frameStart : Nat := 81561
def rule : BoundRule := .sum [.predecessor 0 81659 .coefficient, .predecessor 1 81660 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 81659 .coefficient)
      LeftBound81657.bound (LeftBound81657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81658RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81657.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81657.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 81660 .coefficient)
      LeftBound81638.bound (LeftBound81638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81643RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81638.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81638.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81657.bound, LeftBound81638.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81657.bound, LeftBound81638.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound81657.actual selector witness, LeftBound81638.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81661

namespace LeftBound81674
def owner : Owner := ⟨.program ⟨257⟩, ⟨59101⟩⟩
def transferEvent : Nat := 81674
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 81672 .coefficient, .predecessor 1 81673 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 81672 .coefficient)
      LeftBound81503.bound (LeftBound81503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81671RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81503.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 81673 .coefficient)
      LeftBound81486.bound (LeftBound81486.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81493RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81486.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81486.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81503.bound, LeftBound81486.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81503.bound, LeftBound81486.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound81503.actual selector witness, LeftBound81486.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81674

namespace LeftBound81677
def owner : Owner := ⟨.program ⟨257⟩, ⟨59101⟩⟩
def transferEvent : Nat := 81677
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 81671 .summary, .result 81493 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 81671 .summary)
      LeftBound81505.bound (LeftBound81505.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨57839⟩⟩) (rawTerms := some (Proof.Events319.exact81671RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81505.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 81493 .summary)
      LeftBound81488.bound (LeftBound81488.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59100⟩⟩) (rawTerms := some (Proof.Events318.exact81493RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81488.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81505.bound, LeftBound81488.bound]
def bound : CoeffClass := .finite ⟨32190182365603518530196853751808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81505.bound, LeftBound81488.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound81505.actual selector witness, LeftBound81488.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81677

namespace LeftBound81701
def owner : Owner := ⟨.program ⟨257⟩, ⟨24843⟩⟩
def transferEvent : Nat := 81701
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 81699 .coefficient) (.predecessor 1 81700 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 81699 .coefficient)
      LeftAuthority3361.bound (LeftAuthority3361.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3362RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3361.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3361.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 81700 .coefficient)
      LeftBound75901.bound (LeftBound75901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75901.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75901.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority3361.bound LeftBound75901.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3361.bound, LeftBound75901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority3361.actual selector witness) * (LeftBound75901.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound81701

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
