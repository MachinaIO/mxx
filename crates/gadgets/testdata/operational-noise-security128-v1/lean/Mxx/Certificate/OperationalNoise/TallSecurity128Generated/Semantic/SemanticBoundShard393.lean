import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard378
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard392

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound63303
def owner : Owner := ⟨.program ⟨257⟩, ⟨37942⟩⟩
def transferEvent : Nat := 63303
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 61370 .summary) (.transfer 63302) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 61370 .summary)
      LeftBound61368.bound (LeftBound61368.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨10792⟩⟩) (rawTerms := some (Proof.Events239.exact61370RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 63302)
      LeftBound63302.bound (LeftBound63302.actual selector witness) := by
  exact .transfer (LeftBound63302.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound61368.bound LeftBound63302.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61368.bound, LeftBound63302.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound61368.actual selector witness) * (LeftBound63302.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63303

namespace LeftBound63382
def owner : Owner := ⟨.program ⟨257⟩, ⟨37283⟩⟩
def transferEvent : Nat := 63382
def frameStart : Nat := 63353
def rule : BoundRule := .product (.predecessor 0 63380 .coefficient) (.predecessor 1 63381 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63380 .coefficient)
      LeftAuthority63378.bound (LeftAuthority63378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63379RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63378.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63378.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 63381 .coefficient)
      LeftAuthority63375.bound (LeftAuthority63375.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63376RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63375.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63375.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority63378.bound LeftAuthority63375.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63378.bound, LeftAuthority63375.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority63378.actual selector witness) * (LeftAuthority63375.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63382

namespace LeftBound63386
def owner : Owner := ⟨.program ⟨257⟩, ⟨37284⟩⟩
def transferEvent : Nat := 63386
def frameStart : Nat := 63353
def rule : BoundRule := .identity (.predecessor 0 63385 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63385 .coefficient)
      LeftBound63382.bound (LeftBound63382.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63384RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63382.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63382.derived selector witness)

def rawBound : CoeffClass := LeftBound63382.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63382.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound63382.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound63386

namespace LeftBound63403
def owner : Owner := ⟨.program ⟨257⟩, ⟨38734⟩⟩
def transferEvent : Nat := 63403
def frameStart : Nat := 63353
def rule : BoundRule := .sum [.predecessor 0 63401 .coefficient, .predecessor 1 63402 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63401 .coefficient)
      LeftBound63386.bound (LeftBound63386.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound63386.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 63402 .coefficient)
      LeftAuthority63399.bound (LeftAuthority63399.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority63399.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63386.bound, LeftAuthority63399.bound]
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63386.bound, LeftAuthority63399.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound63386.actual selector witness, LeftAuthority63399.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63403

namespace LeftBound63406
def owner : Owner := ⟨.program ⟨257⟩, ⟨38735⟩⟩
def transferEvent : Nat := 63406
def frameStart : Nat := 63353
def rule : BoundRule := .identity (.predecessor 0 63405 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63405 .coefficient)
      LeftBound63403.bound (LeftBound63403.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound63403.derived selector witness)

def rawBound : CoeffClass := LeftBound63403.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63403.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound63403.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound63406

namespace LeftBound63412
def owner : Owner := ⟨.program ⟨257⟩, ⟨38736⟩⟩
def transferEvent : Nat := 63412
def frameStart : Nat := 63353
def rule : BoundRule := .product (.predecessor 0 63410 .coefficient) (.predecessor 1 63411 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63410 .coefficient)
      LeftAuthority63408.bound (LeftAuthority63408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63408.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63408.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 63411 .coefficient)
      LeftBound63406.bound (LeftBound63406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63407RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63406.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63406.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority63408.bound LeftBound63406.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63408.bound, LeftBound63406.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority63408.actual selector witness) * (LeftBound63406.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63412

namespace LeftBound63428
def owner : Owner := ⟨.program ⟨257⟩, ⟨9554⟩⟩
def transferEvent : Nat := 63428
def frameStart : Nat := 63353
def rule : BoundRule := .scale (.predecessor 0 63426 .coefficient) (.value (.predecessor 1 63427 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63426 .coefficient)
      LeftAuthority63424.bound (LeftAuthority63424.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63425RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63424.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63424.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 63427 .coefficient)
      LeftAuthority63415.bound (LeftAuthority63415.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority63415.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority63424.bound LeftAuthority63415.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63424.bound, LeftAuthority63415.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority63424.actual selector witness) * (LeftAuthority63415.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound63428

namespace LeftBound63431
def owner : Owner := ⟨.program ⟨257⟩, ⟨7298⟩⟩
def transferEvent : Nat := 63431
def frameStart : Nat := 63353
def rule : BoundRule := .identity (.predecessor 0 63430 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63430 .coefficient)
      LeftAuthority63418.bound (LeftAuthority63418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63419RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63418.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63418.derived selector witness)

def rawBound : CoeffClass := LeftAuthority63418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority63418.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound63431

namespace LeftBound63435
def owner : Owner := ⟨.program ⟨257⟩, ⟨9555⟩⟩
def transferEvent : Nat := 63435
def frameStart : Nat := 63353
def rule : BoundRule := .product (.predecessor 0 63433 .coefficient) (.predecessor 1 63434 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63433 .coefficient)
      LeftBound63431.bound (LeftBound63431.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63432RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63431.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63431.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 63434 .coefficient)
      LeftBound63428.bound (LeftBound63428.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63429RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63428.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63428.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound63431.bound LeftBound63428.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63431.bound, LeftBound63428.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound63431.actual selector witness) * (LeftBound63428.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63435

namespace LeftBound63440
def owner : Owner := ⟨.program ⟨257⟩, ⟨38737⟩⟩
def transferEvent : Nat := 63440
def frameStart : Nat := 63353
def rule : BoundRule := .sum [.predecessor 0 63438 .coefficient, .predecessor 1 63439 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63438 .coefficient)
      LeftBound63435.bound (LeftBound63435.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63437RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63435.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63435.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 63439 .coefficient)
      LeftBound63412.bound (LeftBound63412.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63414RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63412.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63412.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63435.bound, LeftBound63412.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63435.bound, LeftBound63412.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound63435.actual selector witness, LeftBound63412.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63440

namespace LeftBound63444
def owner : Owner := ⟨.program ⟨257⟩, ⟨39019⟩⟩
def transferEvent : Nat := 63444
def frameStart : Nat := 63353
def rule : BoundRule := .product (.predecessor 0 63442 .coefficient) (.predecessor 1 63443 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63442 .coefficient)
      LeftBound63440.bound (LeftBound63440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63440.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63440.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 63443 .coefficient)
      LeftAuthority63397.bound (LeftAuthority63397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63397.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63397.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound63440.bound LeftAuthority63397.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63440.bound, LeftAuthority63397.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound63440.actual selector witness) * (LeftAuthority63397.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63444

namespace LeftBound63455
def owner : Owner := ⟨.program ⟨257⟩, ⟨37486⟩⟩
def transferEvent : Nat := 63455
def frameStart : Nat := 63353
def rule : BoundRule := .product (.predecessor 0 63453 .coefficient) (.predecessor 1 63454 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63453 .coefficient)
      LeftAuthority63408.bound (LeftAuthority63408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63408.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63408.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 63454 .coefficient)
      LeftAuthority63451.bound (LeftAuthority63451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63452RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63451.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63451.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority63408.bound LeftAuthority63451.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63408.bound, LeftAuthority63451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority63408.actual selector witness) * (LeftAuthority63451.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63455

namespace LeftBound63463
def owner : Owner := ⟨.program ⟨257⟩, ⟨37487⟩⟩
def transferEvent : Nat := 63463
def frameStart : Nat := 63353
def rule : BoundRule := .sum [.predecessor 0 63461 .coefficient, .predecessor 1 63462 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63461 .coefficient)
      LeftAuthority63459.bound (LeftAuthority63459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63459.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 63462 .coefficient)
      LeftBound63455.bound (LeftBound63455.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63455.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63455.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority63459.bound, LeftBound63455.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63459.bound, LeftBound63455.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority63459.actual selector witness, LeftBound63455.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63463

namespace LeftBound63467
def owner : Owner := ⟨.program ⟨257⟩, ⟨39020⟩⟩
def transferEvent : Nat := 63467
def frameStart : Nat := 63353
def rule : BoundRule := .sum [.predecessor 0 63465 .coefficient, .predecessor 1 63466 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63465 .coefficient)
      LeftBound63463.bound (LeftBound63463.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63464RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63463.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63463.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 63466 .coefficient)
      LeftBound63444.bound (LeftBound63444.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63444.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63444.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63463.bound, LeftBound63444.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63463.bound, LeftBound63444.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound63463.actual selector witness, LeftBound63444.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63467

namespace LeftBound63480
def owner : Owner := ⟨.program ⟨257⟩, ⟨39018⟩⟩
def transferEvent : Nat := 63480
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 63478 .coefficient, .predecessor 1 63479 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 63478 .coefficient)
      LeftBound63301.bound (LeftBound63301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63477RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63301.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63301.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 63479 .coefficient)
      LeftBound63284.bound (LeftBound63284.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63291RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63284.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63284.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63301.bound, LeftBound63284.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63301.bound, LeftBound63284.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound63301.actual selector witness, LeftBound63284.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63480

namespace LeftBound63483
def owner : Owner := ⟨.program ⟨257⟩, ⟨39018⟩⟩
def transferEvent : Nat := 63483
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 63477 .summary, .result 63291 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 63477 .summary)
      LeftBound63303.bound (LeftBound63303.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨37942⟩⟩) (rawTerms := some (Proof.Events247.exact63477RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63303.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 63291 .summary)
      LeftBound63286.bound (LeftBound63286.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39017⟩⟩) (rawTerms := some (Proof.Events247.exact63291RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63286.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63303.bound, LeftBound63286.bound]
def bound : CoeffClass := .finite ⟨2998182198162866044928, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63303.bound, LeftBound63286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound63303.actual selector witness, LeftBound63286.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63483

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
