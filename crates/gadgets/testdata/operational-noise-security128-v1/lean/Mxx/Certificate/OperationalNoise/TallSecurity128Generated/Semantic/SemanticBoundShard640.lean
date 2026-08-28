import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard581
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard639

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound98478
def owner : Owner := ⟨.program ⟨257⟩, ⟨20277⟩⟩
def transferEvent : Nat := 98478
def frameStart : Nat := 98387
def rule : BoundRule := .product (.predecessor 0 98476 .coefficient) (.predecessor 1 98477 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 98476 .coefficient)
      LeftBound98474.bound (LeftBound98474.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98474.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98474.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 98477 .coefficient)
      LeftAuthority98431.bound (LeftAuthority98431.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98432RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98431.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98431.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound98474.bound LeftAuthority98431.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98474.bound, LeftAuthority98431.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound98474.actual selector witness) * (LeftAuthority98431.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98478

namespace LeftBound98489
def owner : Owner := ⟨.program ⟨257⟩, ⟨18630⟩⟩
def transferEvent : Nat := 98489
def frameStart : Nat := 98387
def rule : BoundRule := .product (.predecessor 0 98487 .coefficient) (.predecessor 1 98488 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 98487 .coefficient)
      LeftAuthority98442.bound (LeftAuthority98442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98443RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98442.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98442.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 98488 .coefficient)
      LeftAuthority98485.bound (LeftAuthority98485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98485.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98485.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority98442.bound LeftAuthority98485.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98442.bound, LeftAuthority98485.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority98442.actual selector witness) * (LeftAuthority98485.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98489

namespace LeftBound98497
def owner : Owner := ⟨.program ⟨257⟩, ⟨18631⟩⟩
def transferEvent : Nat := 98497
def frameStart : Nat := 98387
def rule : BoundRule := .sum [.predecessor 0 98495 .coefficient, .predecessor 1 98496 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 98495 .coefficient)
      LeftAuthority98493.bound (LeftAuthority98493.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98493.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 98496 .coefficient)
      LeftBound98489.bound (LeftBound98489.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98491RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98489.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98489.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority98493.bound, LeftBound98489.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98493.bound, LeftBound98489.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority98493.actual selector witness, LeftBound98489.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98497

namespace LeftBound98501
def owner : Owner := ⟨.program ⟨257⟩, ⟨20278⟩⟩
def transferEvent : Nat := 98501
def frameStart : Nat := 98387
def rule : BoundRule := .sum [.predecessor 0 98499 .coefficient, .predecessor 1 98500 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 98499 .coefficient)
      LeftBound98497.bound (LeftBound98497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98497.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98497.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 98500 .coefficient)
      LeftBound98478.bound (LeftBound98478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98478.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98478.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98497.bound, LeftBound98478.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98497.bound, LeftBound98478.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound98497.actual selector witness, LeftBound98478.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98501

namespace LeftBound98514
def owner : Owner := ⟨.program ⟨257⟩, ⟨20276⟩⟩
def transferEvent : Nat := 98514
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 98512 .coefficient, .predecessor 1 98513 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 98512 .coefficient)
      LeftBound98335.bound (LeftBound98335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98511RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98335.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98335.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 98513 .coefficient)
      LeftBound98318.bound (LeftBound98318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98318.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98335.bound, LeftBound98318.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98335.bound, LeftBound98318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound98335.actual selector witness, LeftBound98318.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98514

namespace LeftBound98517
def owner : Owner := ⟨.program ⟨257⟩, ⟨20276⟩⟩
def transferEvent : Nat := 98517
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 98511 .summary, .result 98325 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 98511 .summary)
      LeftBound98337.bound (LeftBound98337.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨19202⟩⟩) (rawTerms := some (Proof.Events384.exact98511RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 98325 .summary)
      LeftBound98320.bound (LeftBound98320.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20275⟩⟩) (rawTerms := some (Proof.Events384.exact98325RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98320.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98337.bound, LeftBound98320.bound]
def bound : CoeffClass := .finite ⟨2997825428629885288448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98337.bound, LeftBound98320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound98337.actual selector witness, LeftBound98320.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98517

namespace LeftBound98521
def owner : Owner := ⟨.program ⟨257⟩, ⟨20809⟩⟩
def transferEvent : Nat := 98521
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 98519 .coefficient) (.predecessor 1 98520 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 98519 .coefficient)
      LeftBound98514.bound (LeftBound98514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98514.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 98520 .coefficient)
      LeftAuthority98240.bound (LeftAuthority98240.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98241RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98240.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98240.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound98514.bound LeftAuthority98240.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98514.bound, LeftAuthority98240.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound98514.actual selector witness) * (LeftAuthority98240.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98521

namespace LeftBound98522
def owner : Owner := ⟨.program ⟨257⟩, ⟨20809⟩⟩
def transferEvent : Nat := 98522
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨20807⟩⟩]⟩ [⟨.result 98241 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 98241 .coefficient)
      LeftAuthority98240.bound (LeftAuthority98240.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨20807⟩⟩) (rawTerms := some (Proof.Events383.exact98241RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98240.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98240.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority98240.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98240.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority98240.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound98522

namespace LeftBound98523
def owner : Owner := ⟨.program ⟨257⟩, ⟨20809⟩⟩
def transferEvent : Nat := 98523
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 98518 .summary) (.transfer 98522) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 98518 .summary)
      LeftBound98517.bound (LeftBound98517.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20276⟩⟩) (rawTerms := some (Proof.Events384.exact98518RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98517.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 98522)
      LeftBound98522.bound (LeftBound98522.actual selector witness) := by
  exact .transfer (LeftBound98522.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound98517.bound LeftBound98522.bound
def bound : CoeffClass := .finite ⟨32188905437706348505289216491520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98517.bound, LeftBound98522.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound98517.actual selector witness) * (LeftBound98522.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98523

namespace LeftBound98534
def owner : Owner := ⟨.program ⟨257⟩, ⟨19558⟩⟩
def transferEvent : Nat := 98534
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 98532 .coefficient) (.value (.predecessor 1 98533 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 98532 .coefficient)
      LeftAuthority98530.bound (LeftAuthority98530.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98531RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98530.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98530.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 98533 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority98530.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98530.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority98530.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound98534

namespace LeftBound98538
def owner : Owner := ⟨.program ⟨257⟩, ⟨19559⟩⟩
def transferEvent : Nat := 98538
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 98536 .coefficient) (.predecessor 1 98537 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 98536 .coefficient)
      LeftBound90617.bound (LeftBound90617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90617.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90617.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 98537 .coefficient)
      LeftBound98534.bound (LeftBound98534.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98535RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98534.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98534.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound90617.bound LeftBound98534.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90617.bound, LeftBound98534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound90617.actual selector witness) * (LeftBound98534.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98538

namespace LeftBound98539
def owner : Owner := ⟨.program ⟨257⟩, ⟨19559⟩⟩
def transferEvent : Nat := 98539
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨19556⟩⟩]⟩ [⟨.result 98531 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 98531 .coefficient)
      LeftAuthority98530.bound (LeftAuthority98530.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨19556⟩⟩) (rawTerms := some (Proof.Events384.exact98531RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98530.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98530.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority98530.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98530.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority98530.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound98539

namespace LeftBound98540
def owner : Owner := ⟨.program ⟨257⟩, ⟨19559⟩⟩
def transferEvent : Nat := 98540
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 90620 .summary) (.transfer 98539) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90620 .summary)
      LeftBound90618.bound (LeftBound90618.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9944⟩⟩) (rawTerms := some (Proof.Events353.exact90620RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 98539)
      LeftBound98539.bound (LeftBound98539.actual selector witness) := by
  exact .transfer (LeftBound98539.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound90618.bound LeftBound98539.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90618.bound, LeftBound98539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound90618.actual selector witness) * (LeftBound98539.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98540

namespace LeftBound98635
def owner : Owner := ⟨.program ⟨257⟩, ⟨18629⟩⟩
def transferEvent : Nat := 98635
def frameStart : Nat := 98596
def rule : BoundRule := .identity (.predecessor 0 98634 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 98634 .coefficient)
      LeftAuthority98632.bound (LeftAuthority98632.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98633RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98632.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98632.derived selector witness)

def rawBound : CoeffClass := LeftAuthority98632.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98632.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority98632.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound98635

namespace LeftBound98652
def owner : Owner := ⟨.program ⟨257⟩, ⟨20086⟩⟩
def transferEvent : Nat := 98652
def frameStart : Nat := 98596
def rule : BoundRule := .sum [.predecessor 0 98650 .coefficient, .predecessor 1 98651 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 98650 .coefficient)
      LeftBound98635.bound (LeftBound98635.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound98635.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 98651 .coefficient)
      LeftAuthority98648.bound (LeftAuthority98648.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority98648.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98635.bound, LeftAuthority98648.bound]
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98635.bound, LeftAuthority98648.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound98635.actual selector witness, LeftAuthority98648.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98652

namespace LeftBound98655
def owner : Owner := ⟨.program ⟨257⟩, ⟨20087⟩⟩
def transferEvent : Nat := 98655
def frameStart : Nat := 98596
def rule : BoundRule := .identity (.predecessor 0 98654 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 98654 .coefficient)
      LeftBound98652.bound (LeftBound98652.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound98652.derived selector witness)

def rawBound : CoeffClass := LeftBound98652.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98652.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound98652.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound98655

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
