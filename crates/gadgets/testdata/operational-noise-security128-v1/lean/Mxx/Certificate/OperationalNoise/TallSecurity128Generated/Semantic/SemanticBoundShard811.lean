import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard098
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard779
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard782
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard784
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard810

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound123452
def owner : Owner := ⟨.program ⟨257⟩, ⟨27079⟩⟩
def transferEvent : Nat := 123452
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 119870 .summary) (.transfer 123451) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119870 .summary)
      LeftBound119868.bound (LeftBound119868.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5527⟩⟩) (rawTerms := some (Proof.Events468.exact119870RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 123451)
      LeftBound123451.bound (LeftBound123451.actual selector witness) := by
  exact .transfer (LeftBound123451.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound119868.bound LeftBound123451.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119868.bound, LeftBound123451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound119868.actual selector witness) * (LeftBound123451.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound123452

namespace LeftBound123547
def owner : Owner := ⟨.program ⟨257⟩, ⟨26377⟩⟩
def transferEvent : Nat := 123547
def frameStart : Nat := 123508
def rule : BoundRule := .identity (.predecessor 0 123546 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123546 .coefficient)
      LeftAuthority123544.bound (LeftAuthority123544.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events482.exact123545RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority123544.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority123544.derived selector witness)

def rawBound : CoeffClass := LeftAuthority123544.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority123544.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority123544.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound123547

namespace LeftBound123564
def owner : Owner := ⟨.program ⟨257⟩, ⟨27750⟩⟩
def transferEvent : Nat := 123564
def frameStart : Nat := 123508
def rule : BoundRule := .sum [.predecessor 0 123562 .coefficient, .predecessor 1 123563 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123562 .coefficient)
      LeftBound123547.bound (LeftBound123547.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound123547.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 123563 .coefficient)
      LeftAuthority123560.bound (LeftAuthority123560.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority123560.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound123547.bound, LeftAuthority123560.bound]
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound123547.bound, LeftAuthority123560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound123547.actual selector witness, LeftAuthority123560.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound123564

namespace LeftBound123567
def owner : Owner := ⟨.program ⟨257⟩, ⟨27751⟩⟩
def transferEvent : Nat := 123567
def frameStart : Nat := 123508
def rule : BoundRule := .identity (.predecessor 0 123566 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123566 .coefficient)
      LeftBound123564.bound (LeftBound123564.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound123564.derived selector witness)

def rawBound : CoeffClass := LeftBound123564.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound123564.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound123564.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound123567

namespace LeftBound123573
def owner : Owner := ⟨.program ⟨257⟩, ⟨27752⟩⟩
def transferEvent : Nat := 123573
def frameStart : Nat := 123508
def rule : BoundRule := .product (.predecessor 0 123571 .coefficient) (.predecessor 1 123572 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123571 .coefficient)
      LeftAuthority123569.bound (LeftAuthority123569.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events482.exact123570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority123569.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority123569.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 123572 .coefficient)
      LeftBound123567.bound (LeftBound123567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events482.exact123568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound123567.bound, RecordedBoundRefines] <;> decide)
      (LeftBound123567.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority123569.bound LeftBound123567.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority123569.bound, LeftBound123567.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority123569.actual selector witness) * (LeftBound123567.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound123573

namespace LeftBound123581
def owner : Owner := ⟨.program ⟨257⟩, ⟨27753⟩⟩
def transferEvent : Nat := 123581
def frameStart : Nat := 123508
def rule : BoundRule := .sum [.predecessor 0 123579 .coefficient, .predecessor 1 123580 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123579 .coefficient)
      LeftAuthority123577.bound (LeftAuthority123577.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events482.exact123578RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority123577.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority123577.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 123580 .coefficient)
      LeftBound123573.bound (LeftBound123573.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events482.exact123575RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound123573.bound, RecordedBoundRefines] <;> decide)
      (LeftBound123573.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority123577.bound, LeftBound123573.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority123577.bound, LeftBound123573.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority123577.actual selector witness, LeftBound123573.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound123581

namespace LeftBound123585
def owner : Owner := ⟨.program ⟨257⟩, ⟨28190⟩⟩
def transferEvent : Nat := 123585
def frameStart : Nat := 123508
def rule : BoundRule := .product (.predecessor 0 123583 .coefficient) (.predecessor 1 123584 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123583 .coefficient)
      LeftBound123581.bound (LeftBound123581.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events482.exact123582RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound123581.bound, RecordedBoundRefines] <;> decide)
      (LeftBound123581.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 123584 .coefficient)
      LeftAuthority123558.bound (LeftAuthority123558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events482.exact123559RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority123558.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority123558.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound123581.bound LeftAuthority123558.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound123581.bound, LeftAuthority123558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound123581.actual selector witness) * (LeftAuthority123558.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound123585

namespace LeftBound123596
def owner : Owner := ⟨.program ⟨257⟩, ⟨26568⟩⟩
def transferEvent : Nat := 123596
def frameStart : Nat := 123508
def rule : BoundRule := .product (.predecessor 0 123594 .coefficient) (.predecessor 1 123595 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123594 .coefficient)
      LeftAuthority123569.bound (LeftAuthority123569.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events482.exact123570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority123569.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority123569.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 123595 .coefficient)
      LeftAuthority123592.bound (LeftAuthority123592.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events482.exact123593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority123592.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority123592.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority123569.bound LeftAuthority123592.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority123569.bound, LeftAuthority123592.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority123569.actual selector witness) * (LeftAuthority123592.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound123596

namespace LeftBound123604
def owner : Owner := ⟨.program ⟨257⟩, ⟨26569⟩⟩
def transferEvent : Nat := 123604
def frameStart : Nat := 123508
def rule : BoundRule := .sum [.predecessor 0 123602 .coefficient, .predecessor 1 123603 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123602 .coefficient)
      LeftAuthority123600.bound (LeftAuthority123600.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events482.exact123601RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority123600.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority123600.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 123603 .coefficient)
      LeftBound123596.bound (LeftBound123596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events482.exact123598RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound123596.bound, RecordedBoundRefines] <;> decide)
      (LeftBound123596.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority123600.bound, LeftBound123596.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority123600.bound, LeftBound123596.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority123600.actual selector witness, LeftBound123596.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound123604

namespace LeftBound123608
def owner : Owner := ⟨.program ⟨257⟩, ⟨28193⟩⟩
def transferEvent : Nat := 123608
def frameStart : Nat := 123508
def rule : BoundRule := .sum [.predecessor 0 123606 .coefficient, .predecessor 1 123607 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123606 .coefficient)
      LeftBound123604.bound (LeftBound123604.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events482.exact123605RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound123604.bound, RecordedBoundRefines] <;> decide)
      (LeftBound123604.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 123607 .coefficient)
      LeftBound123585.bound (LeftBound123585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events482.exact123590RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound123585.bound, RecordedBoundRefines] <;> decide)
      (LeftBound123585.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound123604.bound, LeftBound123585.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound123604.bound, LeftBound123585.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound123604.actual selector witness, LeftBound123585.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound123608

namespace LeftBound123621
def owner : Owner := ⟨.program ⟨257⟩, ⟨28192⟩⟩
def transferEvent : Nat := 123621
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 123619 .coefficient, .predecessor 1 123620 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123619 .coefficient)
      LeftBound123450.bound (LeftBound123450.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events482.exact123618RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound123450.bound, RecordedBoundRefines] <;> decide)
      (LeftBound123450.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 123620 .coefficient)
      LeftBound123433.bound (LeftBound123433.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events482.exact123440RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound123433.bound, RecordedBoundRefines] <;> decide)
      (LeftBound123433.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound123450.bound, LeftBound123433.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound123450.bound, LeftBound123433.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound123450.actual selector witness, LeftBound123433.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound123621

namespace LeftBound123624
def owner : Owner := ⟨.program ⟨257⟩, ⟨28192⟩⟩
def transferEvent : Nat := 123624
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 123618 .summary, .result 123440 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 123618 .summary)
      LeftBound123452.bound (LeftBound123452.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨27079⟩⟩) (rawTerms := some (Proof.Events482.exact123618RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound123452.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 123440 .summary)
      LeftBound123435.bound (LeftBound123435.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28191⟩⟩) (rawTerms := some (Proof.Events482.exact123440RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound123435.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound123452.bound, LeftBound123435.bound]
def bound : CoeffClass := .finite ⟨32191557518723330170883082027008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound123452.bound, LeftBound123435.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound123452.actual selector witness, LeftBound123435.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound123624

namespace LeftBound123648
def owner : Owner := ⟨.program ⟨257⟩, ⟨25683⟩⟩
def transferEvent : Nat := 123648
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 123646 .coefficient) (.predecessor 1 123647 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123646 .coefficient)
      LeftAuthority5513.bound (LeftAuthority5513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5514RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5513.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5513.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 123647 .coefficient)
      LeftBound119776.bound (LeftBound119776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events467.exact119778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119776.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority5513.bound LeftBound119776.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5513.bound, LeftBound119776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority5513.actual selector witness) * (LeftBound119776.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound123648

namespace LeftBound123653
def owner : Owner := ⟨.program ⟨257⟩, ⟨8126⟩⟩
def transferEvent : Nat := 123653
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 123651 .coefficient) (.predecessor 1 123652 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123651 .coefficient)
      LeftBound119647.bound (LeftBound119647.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events467.exact119648RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119647.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119647.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 123652 .coefficient)
      LeftBound21087.bound (LeftBound21087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21087.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound119647.bound LeftBound21087.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119647.bound, LeftBound21087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound119647.actual selector witness) * (LeftBound21087.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound123653

namespace LeftBound123658
def owner : Owner := ⟨.program ⟨257⟩, ⟨25684⟩⟩
def transferEvent : Nat := 123658
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 123656 .coefficient, .predecessor 1 123657 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123656 .coefficient)
      LeftBound123653.bound (LeftBound123653.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events483.exact123655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound123653.bound, RecordedBoundRefines] <;> decide)
      (LeftBound123653.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 123657 .coefficient)
      LeftBound123648.bound (LeftBound123648.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events483.exact123650RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound123648.bound, RecordedBoundRefines] <;> decide)
      (LeftBound123648.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound123653.bound, LeftBound123648.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound123653.bound, LeftBound123648.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound123653.actual selector witness, LeftBound123648.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound123658

namespace LeftBound123662
def owner : Owner := ⟨.program ⟨257⟩, ⟨25685⟩⟩
def transferEvent : Nat := 123662
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 123660 .coefficient, .predecessor 1 123661 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123660 .coefficient)
      LeftBound123658.bound (LeftBound123658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events483.exact123659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound123658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound123658.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 123661 .coefficient)
      LeftBound21079.bound (LeftBound21079.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21079.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21079.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound123658.bound, LeftBound21079.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound123658.bound, LeftBound21079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound123658.actual selector witness, LeftBound21079.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound123662

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
