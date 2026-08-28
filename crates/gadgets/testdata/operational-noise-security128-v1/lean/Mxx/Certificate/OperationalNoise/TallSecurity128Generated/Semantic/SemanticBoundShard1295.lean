import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1291
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1294

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound193481
def owner : Owner := ⟨.program ⟨257⟩, ⟨45932⟩⟩
def transferEvent : Nat := 193481
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨45929⟩⟩]⟩ [⟨.result 193473 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 193473 .coefficient)
      LeftAuthority193472.bound (LeftAuthority193472.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨45929⟩⟩) (rawTerms := some (Proof.Events755.exact193473RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority193472.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority193472.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority193472.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority193472.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority193472.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound193481

namespace LeftBound193482
def owner : Owner := ⟨.program ⟨257⟩, ⟨45932⟩⟩
def transferEvent : Nat := 193482
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 192995 .summary) (.transfer 193481) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 192995 .summary)
      LeftBound192993.bound (LeftBound192993.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5909⟩⟩) (rawTerms := some (Proof.Events753.exact192995RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound192993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 193481)
      LeftBound193481.bound (LeftBound193481.actual selector witness) := by
  exact .transfer (LeftBound193481.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound192993.bound LeftBound193481.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192993.bound, LeftBound193481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound192993.actual selector witness) * (LeftBound193481.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound193482

namespace LeftBound193561
def owner : Owner := ⟨.program ⟨257⟩, ⟨45203⟩⟩
def transferEvent : Nat := 193561
def frameStart : Nat := 193532
def rule : BoundRule := .product (.predecessor 0 193559 .coefficient) (.predecessor 1 193560 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193559 .coefficient)
      LeftAuthority193557.bound (LeftAuthority193557.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events756.exact193558RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority193557.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority193557.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193560 .coefficient)
      LeftAuthority193554.bound (LeftAuthority193554.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events756.exact193555RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority193554.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority193554.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority193557.bound LeftAuthority193554.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority193557.bound, LeftAuthority193554.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority193557.actual selector witness) * (LeftAuthority193554.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound193561

namespace LeftBound193565
def owner : Owner := ⟨.program ⟨257⟩, ⟨45204⟩⟩
def transferEvent : Nat := 193565
def frameStart : Nat := 193532
def rule : BoundRule := .identity (.predecessor 0 193564 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193564 .coefficient)
      LeftBound193561.bound (LeftBound193561.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events756.exact193563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193561.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193561.derived selector witness)

def rawBound : CoeffClass := LeftBound193561.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound193561.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound193561.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound193565

namespace LeftBound193582
def owner : Owner := ⟨.program ⟨257⟩, ⟨46754⟩⟩
def transferEvent : Nat := 193582
def frameStart : Nat := 193532
def rule : BoundRule := .sum [.predecessor 0 193580 .coefficient, .predecessor 1 193581 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193580 .coefficient)
      LeftBound193565.bound (LeftBound193565.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound193565.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193581 .coefficient)
      LeftAuthority193578.bound (LeftAuthority193578.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority193578.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound193565.bound, LeftAuthority193578.bound]
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound193565.bound, LeftAuthority193578.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound193565.actual selector witness, LeftAuthority193578.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound193582

namespace LeftBound193585
def owner : Owner := ⟨.program ⟨257⟩, ⟨46755⟩⟩
def transferEvent : Nat := 193585
def frameStart : Nat := 193532
def rule : BoundRule := .identity (.predecessor 0 193584 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193584 .coefficient)
      LeftBound193582.bound (LeftBound193582.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound193582.derived selector witness)

def rawBound : CoeffClass := LeftBound193582.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound193582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound193582.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound193585

namespace LeftBound193591
def owner : Owner := ⟨.program ⟨257⟩, ⟨46756⟩⟩
def transferEvent : Nat := 193591
def frameStart : Nat := 193532
def rule : BoundRule := .product (.predecessor 0 193589 .coefficient) (.predecessor 1 193590 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193589 .coefficient)
      LeftAuthority193587.bound (LeftAuthority193587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events756.exact193588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority193587.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority193587.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193590 .coefficient)
      LeftBound193585.bound (LeftBound193585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events756.exact193586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193585.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193585.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority193587.bound LeftBound193585.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority193587.bound, LeftBound193585.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority193587.actual selector witness) * (LeftBound193585.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound193591

namespace LeftBound193607
def owner : Owner := ⟨.program ⟨257⟩, ⟨9563⟩⟩
def transferEvent : Nat := 193607
def frameStart : Nat := 193532
def rule : BoundRule := .scale (.predecessor 0 193605 .coefficient) (.value (.predecessor 1 193606 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193605 .coefficient)
      LeftAuthority193603.bound (LeftAuthority193603.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events756.exact193604RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority193603.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority193603.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193606 .coefficient)
      LeftAuthority193594.bound (LeftAuthority193594.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority193594.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority193603.bound LeftAuthority193594.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority193603.bound, LeftAuthority193594.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority193603.actual selector witness) * (LeftAuthority193594.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound193607

namespace LeftBound193610
def owner : Owner := ⟨.program ⟨257⟩, ⟨7301⟩⟩
def transferEvent : Nat := 193610
def frameStart : Nat := 193532
def rule : BoundRule := .identity (.predecessor 0 193609 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193609 .coefficient)
      LeftAuthority193597.bound (LeftAuthority193597.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events756.exact193598RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority193597.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority193597.derived selector witness)

def rawBound : CoeffClass := LeftAuthority193597.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority193597.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority193597.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound193610

namespace LeftBound193614
def owner : Owner := ⟨.program ⟨257⟩, ⟨9564⟩⟩
def transferEvent : Nat := 193614
def frameStart : Nat := 193532
def rule : BoundRule := .product (.predecessor 0 193612 .coefficient) (.predecessor 1 193613 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193612 .coefficient)
      LeftBound193610.bound (LeftBound193610.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events756.exact193611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193610.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193610.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193613 .coefficient)
      LeftBound193607.bound (LeftBound193607.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events756.exact193608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193607.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193607.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound193610.bound LeftBound193607.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound193610.bound, LeftBound193607.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound193610.actual selector witness) * (LeftBound193607.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound193614

namespace LeftBound193619
def owner : Owner := ⟨.program ⟨257⟩, ⟨46757⟩⟩
def transferEvent : Nat := 193619
def frameStart : Nat := 193532
def rule : BoundRule := .sum [.predecessor 0 193617 .coefficient, .predecessor 1 193618 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193617 .coefficient)
      LeftBound193614.bound (LeftBound193614.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events756.exact193616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193614.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193614.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193618 .coefficient)
      LeftBound193591.bound (LeftBound193591.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events756.exact193593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193591.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193591.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound193614.bound, LeftBound193591.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound193614.bound, LeftBound193591.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound193614.actual selector witness, LeftBound193591.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound193619

namespace LeftBound193623
def owner : Owner := ⟨.program ⟨257⟩, ⟨47004⟩⟩
def transferEvent : Nat := 193623
def frameStart : Nat := 193532
def rule : BoundRule := .product (.predecessor 0 193621 .coefficient) (.predecessor 1 193622 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193621 .coefficient)
      LeftBound193619.bound (LeftBound193619.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events756.exact193620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193619.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193619.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193622 .coefficient)
      LeftAuthority193576.bound (LeftAuthority193576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events756.exact193577RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority193576.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority193576.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound193619.bound LeftAuthority193576.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound193619.bound, LeftAuthority193576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound193619.actual selector witness) * (LeftAuthority193576.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound193623

namespace LeftBound193634
def owner : Owner := ⟨.program ⟨257⟩, ⟨45486⟩⟩
def transferEvent : Nat := 193634
def frameStart : Nat := 193532
def rule : BoundRule := .product (.predecessor 0 193632 .coefficient) (.predecessor 1 193633 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193632 .coefficient)
      LeftAuthority193587.bound (LeftAuthority193587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events756.exact193588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority193587.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority193587.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193633 .coefficient)
      LeftAuthority193630.bound (LeftAuthority193630.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events756.exact193631RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority193630.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority193630.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority193587.bound LeftAuthority193630.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority193587.bound, LeftAuthority193630.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority193587.actual selector witness) * (LeftAuthority193630.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound193634

namespace LeftBound193642
def owner : Owner := ⟨.program ⟨257⟩, ⟨45487⟩⟩
def transferEvent : Nat := 193642
def frameStart : Nat := 193532
def rule : BoundRule := .sum [.predecessor 0 193640 .coefficient, .predecessor 1 193641 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193640 .coefficient)
      LeftAuthority193638.bound (LeftAuthority193638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events756.exact193639RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority193638.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority193638.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193641 .coefficient)
      LeftBound193634.bound (LeftBound193634.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events756.exact193636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193634.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193634.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority193638.bound, LeftBound193634.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority193638.bound, LeftBound193634.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority193638.actual selector witness, LeftBound193634.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound193642

namespace LeftBound193646
def owner : Owner := ⟨.program ⟨257⟩, ⟨47005⟩⟩
def transferEvent : Nat := 193646
def frameStart : Nat := 193532
def rule : BoundRule := .sum [.predecessor 0 193644 .coefficient, .predecessor 1 193645 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193644 .coefficient)
      LeftBound193642.bound (LeftBound193642.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events756.exact193643RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193642.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193642.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193645 .coefficient)
      LeftBound193623.bound (LeftBound193623.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events756.exact193628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193623.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193623.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound193642.bound, LeftBound193623.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound193642.bound, LeftBound193623.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound193642.actual selector witness, LeftBound193623.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound193646

namespace LeftBound193659
def owner : Owner := ⟨.program ⟨257⟩, ⟨47003⟩⟩
def transferEvent : Nat := 193659
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 193657 .coefficient, .predecessor 1 193658 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 193657 .coefficient)
      LeftBound193480.bound (LeftBound193480.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events756.exact193656RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193480.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193480.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 193658 .coefficient)
      LeftBound193463.bound (LeftBound193463.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events755.exact193470RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound193463.bound, RecordedBoundRefines] <;> decide)
      (LeftBound193463.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound193480.bound, LeftBound193463.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound193480.bound, LeftBound193463.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound193480.actual selector witness, LeftBound193463.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound193659

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
