import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard068
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard069

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound17371
def owner : Owner := ⟨.program ⟨257⟩, ⟨48724⟩⟩
def transferEvent : Nat := 17371
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 17369 .coefficient) (.value (.predecessor 1 17370 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 17369 .coefficient)
      LeftAuthority17367.bound (LeftAuthority17367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17368RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17367.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 17370 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority17367.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17367.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority17367.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound17371

namespace LeftBound17375
def owner : Owner := ⟨.program ⟨257⟩, ⟨48725⟩⟩
def transferEvent : Nat := 17375
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 17373 .coefficient) (.predecessor 1 17374 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 17373 .coefficient)
      LeftBound17166.bound (LeftBound17166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17169RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17166.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 17374 .coefficient)
      LeftBound17371.bound (LeftBound17371.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17372RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17371.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17371.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound17166.bound LeftBound17371.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17166.bound, LeftBound17371.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound17166.actual selector witness) * (LeftBound17371.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17375

namespace LeftBound17376
def owner : Owner := ⟨.program ⟨257⟩, ⟨48725⟩⟩
def transferEvent : Nat := 17376
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨48722⟩⟩]⟩ [⟨.result 17368 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17368 .coefficient)
      LeftAuthority17367.bound (LeftAuthority17367.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨48722⟩⟩) (rawTerms := some (Proof.Events067.exact17368RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17367.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17367.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority17367.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17367.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority17367.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound17376

namespace LeftBound17377
def owner : Owner := ⟨.program ⟨257⟩, ⟨48725⟩⟩
def transferEvent : Nat := 17377
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 17169 .summary) (.transfer 17376) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17169 .summary)
      LeftBound17167.bound (LeftBound17167.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5443⟩⟩) (rawTerms := some (Proof.Events067.exact17169RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17167.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 17376)
      LeftBound17376.bound (LeftBound17376.actual selector witness) := by
  exact .transfer (LeftBound17376.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound17167.bound LeftBound17376.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17167.bound, LeftBound17376.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound17167.actual selector witness) * (LeftBound17376.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17377

namespace LeftBound17472
def owner : Owner := ⟨.program ⟨257⟩, ⟨48079⟩⟩
def transferEvent : Nat := 17472
def frameStart : Nat := 17433
def rule : BoundRule := .identity (.predecessor 0 17471 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 17471 .coefficient)
      LeftAuthority17469.bound (LeftAuthority17469.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17470RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17469.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17469.derived selector witness)

def rawBound : CoeffClass := LeftAuthority17469.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority17469.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound17472

namespace LeftBound17489
def owner : Owner := ⟨.program ⟨257⟩, ⟨49470⟩⟩
def transferEvent : Nat := 17489
def frameStart : Nat := 17433
def rule : BoundRule := .sum [.predecessor 0 17487 .coefficient, .predecessor 1 17488 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 17487 .coefficient)
      LeftBound17472.bound (LeftBound17472.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound17472.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 17488 .coefficient)
      LeftAuthority17485.bound (LeftAuthority17485.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority17485.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound17472.bound, LeftAuthority17485.bound]
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17472.bound, LeftAuthority17485.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound17472.actual selector witness, LeftAuthority17485.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17489

namespace LeftBound17492
def owner : Owner := ⟨.program ⟨257⟩, ⟨49471⟩⟩
def transferEvent : Nat := 17492
def frameStart : Nat := 17433
def rule : BoundRule := .identity (.predecessor 0 17491 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 17491 .coefficient)
      LeftBound17489.bound (LeftBound17489.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound17489.derived selector witness)

def rawBound : CoeffClass := LeftBound17489.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17489.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound17489.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound17492

namespace LeftBound17498
def owner : Owner := ⟨.program ⟨257⟩, ⟨49472⟩⟩
def transferEvent : Nat := 17498
def frameStart : Nat := 17433
def rule : BoundRule := .product (.predecessor 0 17496 .coefficient) (.predecessor 1 17497 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 17496 .coefficient)
      LeftAuthority17494.bound (LeftAuthority17494.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17494.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17494.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 17497 .coefficient)
      LeftBound17492.bound (LeftBound17492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17493RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17492.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17492.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority17494.bound LeftBound17492.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17494.bound, LeftBound17492.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority17494.actual selector witness) * (LeftBound17492.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17498

namespace LeftBound17506
def owner : Owner := ⟨.program ⟨257⟩, ⟨49473⟩⟩
def transferEvent : Nat := 17506
def frameStart : Nat := 17433
def rule : BoundRule := .sum [.predecessor 0 17504 .coefficient, .predecessor 1 17505 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 17504 .coefficient)
      LeftAuthority17502.bound (LeftAuthority17502.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17502.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17502.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 17505 .coefficient)
      LeftBound17498.bound (LeftBound17498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17498.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17498.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority17502.bound, LeftBound17498.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17502.bound, LeftBound17498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority17502.actual selector witness, LeftBound17498.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17506

namespace LeftBound17510
def owner : Owner := ⟨.program ⟨257⟩, ⟨49812⟩⟩
def transferEvent : Nat := 17510
def frameStart : Nat := 17433
def rule : BoundRule := .product (.predecessor 0 17508 .coefficient) (.predecessor 1 17509 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 17508 .coefficient)
      LeftBound17506.bound (LeftBound17506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17507RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17506.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 17509 .coefficient)
      LeftAuthority17483.bound (LeftAuthority17483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17483.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17483.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound17506.bound LeftAuthority17483.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17506.bound, LeftAuthority17483.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound17506.actual selector witness) * (LeftAuthority17483.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17510

namespace LeftBound17521
def owner : Owner := ⟨.program ⟨257⟩, ⟨48250⟩⟩
def transferEvent : Nat := 17521
def frameStart : Nat := 17433
def rule : BoundRule := .product (.predecessor 0 17519 .coefficient) (.predecessor 1 17520 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 17519 .coefficient)
      LeftAuthority17494.bound (LeftAuthority17494.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17494.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17494.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 17520 .coefficient)
      LeftAuthority17517.bound (LeftAuthority17517.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17517.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17517.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority17494.bound LeftAuthority17517.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17494.bound, LeftAuthority17517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority17494.actual selector witness) * (LeftAuthority17517.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17521

namespace LeftBound17529
def owner : Owner := ⟨.program ⟨257⟩, ⟨48251⟩⟩
def transferEvent : Nat := 17529
def frameStart : Nat := 17433
def rule : BoundRule := .sum [.predecessor 0 17527 .coefficient, .predecessor 1 17528 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 17527 .coefficient)
      LeftAuthority17525.bound (LeftAuthority17525.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17526RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17525.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17525.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 17528 .coefficient)
      LeftBound17521.bound (LeftBound17521.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17521.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17521.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority17525.bound, LeftBound17521.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17525.bound, LeftBound17521.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority17525.actual selector witness, LeftBound17521.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17529

namespace LeftBound17533
def owner : Owner := ⟨.program ⟨257⟩, ⟨49815⟩⟩
def transferEvent : Nat := 17533
def frameStart : Nat := 17433
def rule : BoundRule := .sum [.predecessor 0 17531 .coefficient, .predecessor 1 17532 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 17531 .coefficient)
      LeftBound17529.bound (LeftBound17529.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17529.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17529.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 17532 .coefficient)
      LeftBound17510.bound (LeftBound17510.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17510.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17510.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound17529.bound, LeftBound17510.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17529.bound, LeftBound17510.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound17529.actual selector witness, LeftBound17510.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17533

namespace LeftBound17546
def owner : Owner := ⟨.program ⟨257⟩, ⟨49814⟩⟩
def transferEvent : Nat := 17546
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 17544 .coefficient, .predecessor 1 17545 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 17544 .coefficient)
      LeftBound17375.bound (LeftBound17375.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17375.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17375.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 17545 .coefficient)
      LeftBound17358.bound (LeftBound17358.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17365RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17358.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17358.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound17375.bound, LeftBound17358.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17375.bound, LeftBound17358.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound17375.actual selector witness, LeftBound17358.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17546

namespace LeftBound17549
def owner : Owner := ⟨.program ⟨257⟩, ⟨49814⟩⟩
def transferEvent : Nat := 17549
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 17543 .summary, .result 17365 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17543 .summary)
      LeftBound17377.bound (LeftBound17377.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨48725⟩⟩) (rawTerms := some (Proof.Events068.exact17543RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17377.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17365 .summary)
      LeftBound17360.bound (LeftBound17360.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49813⟩⟩) (rawTerms := some (Proof.Events067.exact17365RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17360.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound17377.bound, LeftBound17360.bound]
def bound : CoeffClass := .finite ⟨32194504275408640829496428331008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17377.bound, LeftBound17360.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound17377.actual selector witness, LeftBound17360.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17549

namespace LeftBound17572
def owner : Owner := ⟨.program ⟨257⟩, ⟨110⟩⟩
def transferEvent : Nat := 17572
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 17571 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 17571 .coefficient)
      LeftAuthority17048.bound (LeftAuthority17048.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events066.exact17049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17048.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17048.derived selector witness)

def rawBound : CoeffClass := LeftAuthority17048.bound
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17048.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority17048.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound17572

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
