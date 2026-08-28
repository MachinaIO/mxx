import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard071
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2000

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound295571
def owner : Owner := ⟨.program ⟨257⟩, ⟨44920⟩⟩
def transferEvent : Nat := 295571
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 295569 .coefficient) (.predecessor 1 295570 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 295569 .coefficient)
      LeftBound295565.bound (LeftBound295565.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1154.exact295568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound295565.bound, RecordedBoundRefines] <;> decide)
      (LeftBound295565.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 295570 .coefficient)
      LeftAuthority14315.bound (LeftAuthority14315.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14316RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14315.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14315.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound295565.bound LeftAuthority14315.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound295565.bound, LeftAuthority14315.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound295565.actual selector witness) * (LeftAuthority14315.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound295571

namespace LeftBound295572
def owner : Owner := ⟨.program ⟨257⟩, ⟨44920⟩⟩
def transferEvent : Nat := 295572
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨14631⟩⟩], []⟩ [⟨.result 14316 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 14316 .coefficient)
      LeftAuthority14315.bound (LeftAuthority14315.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨14631⟩⟩) (rawTerms := some (Proof.Events055.exact14316RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14315.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14315.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority14315.bound []
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14315.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority14315.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound295572

namespace LeftBound295573
def owner : Owner := ⟨.program ⟨257⟩, ⟨44920⟩⟩
def transferEvent : Nat := 295573
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 295568 .summary) (.transfer 295572) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 295568 .summary)
      LeftBound295566.bound (LeftBound295566.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44919⟩⟩) (rawTerms := some (Proof.Events1154.exact295568RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound295566.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 295572)
      LeftBound295572.bound (LeftBound295572.actual selector witness) := by
  exact .transfer (LeftBound295572.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound295566.bound LeftBound295572.bound
def bound : CoeffClass := .finite ⟨49414144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound295566.bound, LeftBound295572.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound295566.actual selector witness) * (LeftBound295572.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound295573

namespace LeftBound295579
def owner : Owner := ⟨.program ⟨257⟩, ⟨14632⟩⟩
def transferEvent : Nat := 295579
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 295577 .coefficient) (.predecessor 1 295578 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 295577 .coefficient)
      LeftAuthority14315.bound (LeftAuthority14315.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14316RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14315.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14315.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 295578 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority14315.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14315.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority14315.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound295579

namespace LeftBound295584
def owner : Owner := ⟨.program ⟨257⟩, ⟨7449⟩⟩
def transferEvent : Nat := 295584
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 295582 .coefficient) (.predecessor 1 295583 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 295582 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 295583 .coefficient)
      LeftBound17621.bound (LeftBound17621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17622RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17621.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17621.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound26.bound LeftBound17621.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound17621.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound26.actual selector witness) * (LeftBound17621.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound295584

namespace LeftBound295589
def owner : Owner := ⟨.program ⟨257⟩, ⟨14633⟩⟩
def transferEvent : Nat := 295589
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 295587 .coefficient, .predecessor 1 295588 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 295587 .coefficient)
      LeftBound295584.bound (LeftBound295584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1154.exact295586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound295584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound295584.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 295588 .coefficient)
      LeftBound295579.bound (LeftBound295579.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1154.exact295581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound295579.bound, RecordedBoundRefines] <;> decide)
      (LeftBound295579.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound295584.bound, LeftBound295579.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound295584.bound, LeftBound295579.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound295584.actual selector witness, LeftBound295579.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound295589

namespace LeftBound295593
def owner : Owner := ⟨.program ⟨257⟩, ⟨14634⟩⟩
def transferEvent : Nat := 295593
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 295591 .coefficient, .predecessor 1 295592 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 295591 .coefficient)
      LeftBound295589.bound (LeftBound295589.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1154.exact295590RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound295589.bound, RecordedBoundRefines] <;> decide)
      (LeftBound295589.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 295592 .coefficient)
      LeftBound17613.bound (LeftBound17613.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17614RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17613.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17613.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound295589.bound, LeftBound17613.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound295589.bound, LeftBound17613.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound295589.actual selector witness, LeftBound17613.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound295593

namespace LeftBound295594
def owner : Owner := ⟨.program ⟨257⟩, ⟨14634⟩⟩
def transferEvent : Nat := 295594
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨127⟩⟩]⟩ [⟨.result 17614 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17614 .coefficient)
      LeftBound17613.bound (LeftBound17613.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨127⟩⟩) (rawTerms := some (Proof.Events068.exact17614RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17613.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17613.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound17613.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17613.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound17613.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound295594

namespace LeftBound295599
def owner : Owner := ⟨.program ⟨257⟩, ⟨14635⟩⟩
def transferEvent : Nat := 295599
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 295597 .coefficient) (.predecessor 1 295598 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 295597 .coefficient)
      LeftBound295593.bound (LeftBound295593.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1154.exact295596RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound295593.bound, RecordedBoundRefines] <;> decide)
      (LeftBound295593.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 295598 .coefficient)
      LeftBound17610.bound (LeftBound17610.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17610.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17610.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound295593.bound LeftBound17610.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound295593.bound, LeftBound17610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound295593.actual selector witness) * (LeftBound17610.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound295599

namespace LeftBound295600
def owner : Owner := ⟨.program ⟨257⟩, ⟨14635⟩⟩
def transferEvent : Nat := 295600
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩ [⟨.result 17607 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17607 .coefficient)
      LeftAuthority17606.bound (LeftAuthority17606.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9562⟩⟩) (rawTerms := some (Proof.Events068.exact17607RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17606.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17606.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority17606.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17606.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority17606.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound295600

namespace LeftBound295601
def owner : Owner := ⟨.program ⟨257⟩, ⟨14635⟩⟩
def transferEvent : Nat := 295601
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 295596 .summary) (.transfer 295600) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 295596 .summary)
      LeftBound295594.bound (LeftBound295594.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨14634⟩⟩) (rawTerms := some (Proof.Events1154.exact295596RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound295594.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 295600)
      LeftBound295600.bound (LeftBound295600.actual selector witness) := by
  exact .transfer (LeftBound295600.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound295594.bound LeftBound295600.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound295594.bound, LeftBound295600.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound295594.actual selector witness) * (LeftBound295600.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound295601

namespace LeftBound295609
def owner : Owner := ⟨.program ⟨257⟩, ⟨44921⟩⟩
def transferEvent : Nat := 295609
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 295607 .coefficient, .predecessor 1 295608 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 295607 .coefficient)
      LeftBound295599.bound (LeftBound295599.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1154.exact295606RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound295599.bound, RecordedBoundRefines] <;> decide)
      (LeftBound295599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 295608 .coefficient)
      LeftBound295571.bound (LeftBound295571.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1154.exact295576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound295571.bound, RecordedBoundRefines] <;> decide)
      (LeftBound295571.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound295599.bound, LeftBound295571.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound295599.bound, LeftBound295571.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound295599.actual selector witness, LeftBound295571.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound295609

namespace LeftBound295611
def owner : Owner := ⟨.program ⟨257⟩, ⟨44921⟩⟩
def transferEvent : Nat := 295611
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 295606 .summary, .result 295576 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 295606 .summary)
      LeftBound295601.bound (LeftBound295601.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨14635⟩⟩) (rawTerms := some (Proof.Events1154.exact295606RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound295601.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 295576 .summary)
      LeftBound295573.bound (LeftBound295573.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44920⟩⟩) (rawTerms := some (Proof.Events1154.exact295576RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound295573.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound295601.bound, LeftBound295573.bound]
def bound : CoeffClass := .finite ⟨279222288384, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound295601.bound, LeftBound295573.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound295601.actual selector witness, LeftBound295573.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound295611

namespace LeftBound295615
def owner : Owner := ⟨.program ⟨257⟩, ⟨46870⟩⟩
def transferEvent : Nat := 295615
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 295613 .coefficient) (.predecessor 1 295614 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 295613 .coefficient)
      LeftBound295609.bound (LeftBound295609.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1154.exact295612RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound295609.bound, RecordedBoundRefines] <;> decide)
      (LeftBound295609.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 295614 .coefficient)
      LeftAuthority295547.bound (LeftAuthority295547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1154.exact295548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority295547.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority295547.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound295609.bound LeftAuthority295547.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound295609.bound, LeftAuthority295547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound295609.actual selector witness) * (LeftAuthority295547.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound295615

namespace LeftBound295616
def owner : Owner := ⟨.program ⟨257⟩, ⟨46870⟩⟩
def transferEvent : Nat := 295616
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨46869⟩⟩]⟩ [⟨.result 295548 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 295548 .coefficient)
      LeftAuthority295547.bound (LeftAuthority295547.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨46869⟩⟩) (rawTerms := some (Proof.Events1154.exact295548RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority295547.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority295547.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority295547.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority295547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority295547.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound295616

namespace LeftBound295617
def owner : Owner := ⟨.program ⟨257⟩, ⟨46870⟩⟩
def transferEvent : Nat := 295617
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 295612 .summary) (.transfer 295616) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 295612 .summary)
      LeftBound295611.bound (LeftBound295611.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44921⟩⟩) (rawTerms := some (Proof.Events1154.exact295612RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound295611.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 295616)
      LeftBound295616.bound (LeftBound295616.actual selector witness) := by
  exact .transfer (LeftBound295616.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound295611.bound LeftBound295616.bound
def bound : CoeffClass := .finite ⟨2998126492308901724160, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound295611.bound, LeftBound295616.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound295611.actual selector witness) * (LeftBound295616.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound295617

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
