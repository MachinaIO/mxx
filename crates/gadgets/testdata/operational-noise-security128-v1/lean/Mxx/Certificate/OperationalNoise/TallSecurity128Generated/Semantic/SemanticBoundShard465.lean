import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard378
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard426
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard464

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound74343
def owner : Owner := ⟨.program ⟨257⟩, ⟨53164⟩⟩
def transferEvent : Nat := 74343
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 67822 .summary) (.transfer 74342) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 67822 .summary)
      LeftBound67821.bound (LeftBound67821.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52598⟩⟩) (rawTerms := some (Proof.Events264.exact67822RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67821.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 74342)
      LeftBound74342.bound (LeftBound74342.actual selector witness) := by
  exact .transfer (LeftBound74342.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound67821.bound LeftBound74342.bound
def bound : CoeffClass := .finite ⟨32189593014266254325632330629120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67821.bound, LeftBound74342.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound67821.actual selector witness) * (LeftBound74342.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound74343

namespace LeftBound74354
def owner : Owner := ⟨.program ⟨257⟩, ⟨51894⟩⟩
def transferEvent : Nat := 74354
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 74352 .coefficient) (.value (.predecessor 1 74353 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 74352 .coefficient)
      LeftAuthority74350.bound (LeftAuthority74350.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events290.exact74351RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority74350.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority74350.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 74353 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority74350.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority74350.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority74350.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound74354

namespace LeftBound74358
def owner : Owner := ⟨.program ⟨257⟩, ⟨51895⟩⟩
def transferEvent : Nat := 74358
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 74356 .coefficient) (.predecessor 1 74357 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 74356 .coefficient)
      LeftBound61367.bound (LeftBound61367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 74357 .coefficient)
      LeftBound74354.bound (LeftBound74354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events290.exact74355RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74354.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound61367.bound LeftBound74354.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61367.bound, LeftBound74354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound61367.actual selector witness) * (LeftBound74354.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound74358

namespace LeftBound74359
def owner : Owner := ⟨.program ⟨257⟩, ⟨51895⟩⟩
def transferEvent : Nat := 74359
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨51892⟩⟩]⟩ [⟨.result 74351 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 74351 .coefficient)
      LeftAuthority74350.bound (LeftAuthority74350.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨51892⟩⟩) (rawTerms := some (Proof.Events290.exact74351RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority74350.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority74350.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority74350.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority74350.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority74350.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound74359

namespace LeftBound74360
def owner : Owner := ⟨.program ⟨257⟩, ⟨51895⟩⟩
def transferEvent : Nat := 74360
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 61370 .summary) (.transfer 74359) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 61370 .summary)
      LeftBound61368.bound (LeftBound61368.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨10792⟩⟩) (rawTerms := some (Proof.Events239.exact61370RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 74359)
      LeftBound74359.bound (LeftBound74359.actual selector witness) := by
  exact .transfer (LeftBound74359.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound61368.bound LeftBound74359.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61368.bound, LeftBound74359.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound61368.actual selector witness) * (LeftBound74359.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound74360

namespace LeftBound74455
def owner : Owner := ⟨.program ⟨257⟩, ⟨50945⟩⟩
def transferEvent : Nat := 74455
def frameStart : Nat := 74416
def rule : BoundRule := .identity (.predecessor 0 74454 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 74454 .coefficient)
      LeftAuthority74452.bound (LeftAuthority74452.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events290.exact74453RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority74452.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority74452.derived selector witness)

def rawBound : CoeffClass := LeftAuthority74452.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority74452.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority74452.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound74455

namespace LeftBound74472
def owner : Owner := ⟨.program ⟨257⟩, ⟨52394⟩⟩
def transferEvent : Nat := 74472
def frameStart : Nat := 74416
def rule : BoundRule := .sum [.predecessor 0 74470 .coefficient, .predecessor 1 74471 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 74470 .coefficient)
      LeftBound74455.bound (LeftBound74455.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound74455.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 74471 .coefficient)
      LeftAuthority74468.bound (LeftAuthority74468.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority74468.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74455.bound, LeftAuthority74468.bound]
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74455.bound, LeftAuthority74468.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound74455.actual selector witness, LeftAuthority74468.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74472

namespace LeftBound74475
def owner : Owner := ⟨.program ⟨257⟩, ⟨52395⟩⟩
def transferEvent : Nat := 74475
def frameStart : Nat := 74416
def rule : BoundRule := .identity (.predecessor 0 74474 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 74474 .coefficient)
      LeftBound74472.bound (LeftBound74472.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound74472.derived selector witness)

def rawBound : CoeffClass := LeftBound74472.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74472.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound74472.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound74475

namespace LeftBound74481
def owner : Owner := ⟨.program ⟨257⟩, ⟨52396⟩⟩
def transferEvent : Nat := 74481
def frameStart : Nat := 74416
def rule : BoundRule := .product (.predecessor 0 74479 .coefficient) (.predecessor 1 74480 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 74479 .coefficient)
      LeftAuthority74477.bound (LeftAuthority74477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events290.exact74478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority74477.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority74477.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 74480 .coefficient)
      LeftBound74475.bound (LeftBound74475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events290.exact74476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74475.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority74477.bound LeftBound74475.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority74477.bound, LeftBound74475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority74477.actual selector witness) * (LeftBound74475.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound74481

namespace LeftBound74489
def owner : Owner := ⟨.program ⟨257⟩, ⟨52397⟩⟩
def transferEvent : Nat := 74489
def frameStart : Nat := 74416
def rule : BoundRule := .sum [.predecessor 0 74487 .coefficient, .predecessor 1 74488 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 74487 .coefficient)
      LeftAuthority74485.bound (LeftAuthority74485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events290.exact74486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority74485.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority74485.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 74488 .coefficient)
      LeftBound74481.bound (LeftBound74481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events290.exact74483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74481.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74481.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority74485.bound, LeftBound74481.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority74485.bound, LeftBound74481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority74485.actual selector witness, LeftBound74481.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74489

namespace LeftBound74493
def owner : Owner := ⟨.program ⟨257⟩, ⟨53163⟩⟩
def transferEvent : Nat := 74493
def frameStart : Nat := 74416
def rule : BoundRule := .product (.predecessor 0 74491 .coefficient) (.predecessor 1 74492 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 74491 .coefficient)
      LeftBound74489.bound (LeftBound74489.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events290.exact74490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74489.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74489.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 74492 .coefficient)
      LeftAuthority74466.bound (LeftAuthority74466.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events290.exact74467RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority74466.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority74466.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound74489.bound LeftAuthority74466.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74489.bound, LeftAuthority74466.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound74489.actual selector witness) * (LeftAuthority74466.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound74493

namespace LeftBound74504
def owner : Owner := ⟨.program ⟨257⟩, ⟨51301⟩⟩
def transferEvent : Nat := 74504
def frameStart : Nat := 74416
def rule : BoundRule := .product (.predecessor 0 74502 .coefficient) (.predecessor 1 74503 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 74502 .coefficient)
      LeftAuthority74477.bound (LeftAuthority74477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events290.exact74478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority74477.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority74477.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 74503 .coefficient)
      LeftAuthority74500.bound (LeftAuthority74500.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events291.exact74501RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority74500.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority74500.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority74477.bound LeftAuthority74500.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority74477.bound, LeftAuthority74500.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority74477.actual selector witness) * (LeftAuthority74500.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound74504

namespace LeftBound74512
def owner : Owner := ⟨.program ⟨257⟩, ⟨51302⟩⟩
def transferEvent : Nat := 74512
def frameStart : Nat := 74416
def rule : BoundRule := .sum [.predecessor 0 74510 .coefficient, .predecessor 1 74511 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 74510 .coefficient)
      LeftAuthority74508.bound (LeftAuthority74508.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events291.exact74509RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority74508.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority74508.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 74511 .coefficient)
      LeftBound74504.bound (LeftBound74504.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events291.exact74506RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74504.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74504.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority74508.bound, LeftBound74504.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority74508.bound, LeftBound74504.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority74508.actual selector witness, LeftBound74504.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74512

namespace LeftBound74516
def owner : Owner := ⟨.program ⟨257⟩, ⟨53168⟩⟩
def transferEvent : Nat := 74516
def frameStart : Nat := 74416
def rule : BoundRule := .sum [.predecessor 0 74514 .coefficient, .predecessor 1 74515 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 74514 .coefficient)
      LeftBound74512.bound (LeftBound74512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events291.exact74513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74512.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 74515 .coefficient)
      LeftBound74493.bound (LeftBound74493.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events291.exact74498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74493.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74493.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74512.bound, LeftBound74493.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74512.bound, LeftBound74493.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound74512.actual selector witness, LeftBound74493.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74516

namespace LeftBound74529
def owner : Owner := ⟨.program ⟨257⟩, ⟨53165⟩⟩
def transferEvent : Nat := 74529
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 74527 .coefficient, .predecessor 1 74528 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 74527 .coefficient)
      LeftBound74358.bound (LeftBound74358.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events291.exact74526RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74358.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74358.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 74528 .coefficient)
      LeftBound74341.bound (LeftBound74341.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events290.exact74348RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74341.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74341.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74358.bound, LeftBound74341.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74358.bound, LeftBound74341.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound74358.actual selector witness, LeftBound74341.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74529

namespace LeftBound74532
def owner : Owner := ⟨.program ⟨257⟩, ⟨53165⟩⟩
def transferEvent : Nat := 74532
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 74526 .summary, .result 74348 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 74526 .summary)
      LeftBound74360.bound (LeftBound74360.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨51895⟩⟩) (rawTerms := some (Proof.Events291.exact74526RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound74360.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 74348 .summary)
      LeftBound74343.bound (LeftBound74343.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53164⟩⟩) (rawTerms := some (Proof.Events290.exact74348RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound74343.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74360.bound, LeftBound74343.bound]
def bound : CoeffClass := .finite ⟨32189593014266456398474184491008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74360.bound, LeftBound74343.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound74360.actual selector witness, LeftBound74343.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74532

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
