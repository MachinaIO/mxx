import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard327

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound53598
def owner : Owner := ⟨.program ⟨257⟩, ⟨33258⟩⟩
def transferEvent : Nat := 53598
def frameStart : Nat := 53548
def rule : BoundRule := .sum [.predecessor 0 53596 .coefficient, .predecessor 1 53597 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 53596 .coefficient)
      LeftBound53581.bound (LeftBound53581.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound53581.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 53597 .coefficient)
      LeftAuthority53594.bound (LeftAuthority53594.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority53594.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53581.bound, LeftAuthority53594.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53581.bound, LeftAuthority53594.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound53581.actual selector witness, LeftAuthority53594.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53598

namespace LeftBound53601
def owner : Owner := ⟨.program ⟨257⟩, ⟨33259⟩⟩
def transferEvent : Nat := 53601
def frameStart : Nat := 53548
def rule : BoundRule := .identity (.predecessor 0 53600 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 53600 .coefficient)
      LeftBound53598.bound (LeftBound53598.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound53598.derived selector witness)

def rawBound : CoeffClass := LeftBound53598.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53598.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound53598.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound53601

namespace LeftBound53607
def owner : Owner := ⟨.program ⟨257⟩, ⟨33260⟩⟩
def transferEvent : Nat := 53607
def frameStart : Nat := 53548
def rule : BoundRule := .product (.predecessor 0 53605 .coefficient) (.predecessor 1 53606 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 53605 .coefficient)
      LeftAuthority53603.bound (LeftAuthority53603.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53604RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53603.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53603.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 53606 .coefficient)
      LeftBound53601.bound (LeftBound53601.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53602RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53601.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53601.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority53603.bound LeftBound53601.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53603.bound, LeftBound53601.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority53603.actual selector witness) * (LeftBound53601.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53607

namespace LeftBound53623
def owner : Owner := ⟨.program ⟨257⟩, ⟨9578⟩⟩
def transferEvent : Nat := 53623
def frameStart : Nat := 53548
def rule : BoundRule := .scale (.predecessor 0 53621 .coefficient) (.value (.predecessor 1 53622 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 53621 .coefficient)
      LeftAuthority53619.bound (LeftAuthority53619.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53619.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53619.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 53622 .coefficient)
      LeftAuthority53610.bound (LeftAuthority53610.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority53610.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority53619.bound LeftAuthority53610.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53619.bound, LeftAuthority53610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority53619.actual selector witness) * (LeftAuthority53610.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound53623

namespace LeftBound53626
def owner : Owner := ⟨.program ⟨257⟩, ⟨7287⟩⟩
def transferEvent : Nat := 53626
def frameStart : Nat := 53548
def rule : BoundRule := .identity (.predecessor 0 53625 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 53625 .coefficient)
      LeftAuthority53613.bound (LeftAuthority53613.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53614RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53613.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53613.derived selector witness)

def rawBound : CoeffClass := LeftAuthority53613.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53613.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority53613.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound53626

namespace LeftBound53630
def owner : Owner := ⟨.program ⟨257⟩, ⟨9579⟩⟩
def transferEvent : Nat := 53630
def frameStart : Nat := 53548
def rule : BoundRule := .product (.predecessor 0 53628 .coefficient) (.predecessor 1 53629 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 53628 .coefficient)
      LeftBound53626.bound (LeftBound53626.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53627RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53626.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53626.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 53629 .coefficient)
      LeftBound53623.bound (LeftBound53623.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53624RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53623.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53623.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound53626.bound LeftBound53623.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53626.bound, LeftBound53623.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound53626.actual selector witness) * (LeftBound53623.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53630

namespace LeftBound53635
def owner : Owner := ⟨.program ⟨257⟩, ⟨33261⟩⟩
def transferEvent : Nat := 53635
def frameStart : Nat := 53548
def rule : BoundRule := .sum [.predecessor 0 53633 .coefficient, .predecessor 1 53634 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 53633 .coefficient)
      LeftBound53630.bound (LeftBound53630.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53632RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53630.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53630.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 53634 .coefficient)
      LeftBound53607.bound (LeftBound53607.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53609RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53607.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53607.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53630.bound, LeftBound53607.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53630.bound, LeftBound53607.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound53630.actual selector witness, LeftBound53607.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53635

namespace LeftBound53639
def owner : Owner := ⟨.program ⟨257⟩, ⟨33550⟩⟩
def transferEvent : Nat := 53639
def frameStart : Nat := 53548
def rule : BoundRule := .product (.predecessor 0 53637 .coefficient) (.predecessor 1 53638 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 53637 .coefficient)
      LeftBound53635.bound (LeftBound53635.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53635.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53635.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 53638 .coefficient)
      LeftAuthority53592.bound (LeftAuthority53592.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53592.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53592.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound53635.bound LeftAuthority53592.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53635.bound, LeftAuthority53592.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound53635.actual selector witness) * (LeftAuthority53592.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53639

namespace LeftBound53650
def owner : Owner := ⟨.program ⟨257⟩, ⟨31894⟩⟩
def transferEvent : Nat := 53650
def frameStart : Nat := 53548
def rule : BoundRule := .product (.predecessor 0 53648 .coefficient) (.predecessor 1 53649 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 53648 .coefficient)
      LeftAuthority53603.bound (LeftAuthority53603.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53604RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53603.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53603.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 53649 .coefficient)
      LeftAuthority53646.bound (LeftAuthority53646.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53647RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53646.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53646.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority53603.bound LeftAuthority53646.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53603.bound, LeftAuthority53646.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority53603.actual selector witness) * (LeftAuthority53646.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53650

namespace LeftBound53658
def owner : Owner := ⟨.program ⟨257⟩, ⟨31895⟩⟩
def transferEvent : Nat := 53658
def frameStart : Nat := 53548
def rule : BoundRule := .sum [.predecessor 0 53656 .coefficient, .predecessor 1 53657 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 53656 .coefficient)
      LeftAuthority53654.bound (LeftAuthority53654.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53654.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53654.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 53657 .coefficient)
      LeftBound53650.bound (LeftBound53650.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53652RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53650.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53650.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority53654.bound, LeftBound53650.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53654.bound, LeftBound53650.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority53654.actual selector witness, LeftBound53650.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53658

namespace LeftBound53662
def owner : Owner := ⟨.program ⟨257⟩, ⟨33551⟩⟩
def transferEvent : Nat := 53662
def frameStart : Nat := 53548
def rule : BoundRule := .sum [.predecessor 0 53660 .coefficient, .predecessor 1 53661 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 53660 .coefficient)
      LeftBound53658.bound (LeftBound53658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53658.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 53661 .coefficient)
      LeftBound53639.bound (LeftBound53639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53644RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53639.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53639.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53658.bound, LeftBound53639.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53658.bound, LeftBound53639.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound53658.actual selector witness, LeftBound53639.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53662

namespace LeftBound53675
def owner : Owner := ⟨.program ⟨257⟩, ⟨33549⟩⟩
def transferEvent : Nat := 53675
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 53673 .coefficient, .predecessor 1 53674 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 53673 .coefficient)
      LeftBound53496.bound (LeftBound53496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53672RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53496.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53496.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 53674 .coefficient)
      LeftBound53479.bound (LeftBound53479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53479.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53496.bound, LeftBound53479.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53496.bound, LeftBound53479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound53496.actual selector witness, LeftBound53479.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53675

namespace LeftBound53678
def owner : Owner := ⟨.program ⟨257⟩, ⟨33549⟩⟩
def transferEvent : Nat := 53678
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 53672 .summary, .result 53486 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 53672 .summary)
      LeftBound53498.bound (LeftBound53498.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨32472⟩⟩) (rawTerms := some (Proof.Events209.exact53672RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53498.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 53486 .summary)
      LeftBound53481.bound (LeftBound53481.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33548⟩⟩) (rawTerms := some (Proof.Events208.exact53486RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53481.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53498.bound, LeftBound53481.bound]
def bound : CoeffClass := .finite ⟨2997852872440114577408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53498.bound, LeftBound53481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound53498.actual selector witness, LeftBound53481.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53678

namespace LeftBound53682
def owner : Owner := ⟨.program ⟨257⟩, ⟨34142⟩⟩
def transferEvent : Nat := 53682
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 53680 .coefficient) (.predecessor 1 53681 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 53680 .coefficient)
      LeftBound53675.bound (LeftBound53675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53679RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53675.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53675.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 53681 .coefficient)
      LeftAuthority53401.bound (LeftAuthority53401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53401.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53401.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound53675.bound LeftAuthority53401.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53675.bound, LeftAuthority53401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound53675.actual selector witness) * (LeftAuthority53401.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53682

namespace LeftBound53683
def owner : Owner := ⟨.program ⟨257⟩, ⟨34142⟩⟩
def transferEvent : Nat := 53683
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨34140⟩⟩]⟩ [⟨.result 53402 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 53402 .coefficient)
      LeftAuthority53401.bound (LeftAuthority53401.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨34140⟩⟩) (rawTerms := some (Proof.Events208.exact53402RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53401.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53401.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority53401.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority53401.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound53683

namespace LeftBound53684
def owner : Owner := ⟨.program ⟨257⟩, ⟨34142⟩⟩
def transferEvent : Nat := 53684
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 53679 .summary) (.transfer 53683) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 53679 .summary)
      LeftBound53678.bound (LeftBound53678.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33549⟩⟩) (rawTerms := some (Proof.Events209.exact53679RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53678.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 53683)
      LeftBound53683.bound (LeftBound53683.actual selector witness) := by
  exact .transfer (LeftBound53683.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound53678.bound LeftBound53683.bound
def bound : CoeffClass := .finite ⟨32189200113374879571150551121920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53678.bound, LeftBound53683.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound53678.actual selector witness) * (LeftBound53683.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53684

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
