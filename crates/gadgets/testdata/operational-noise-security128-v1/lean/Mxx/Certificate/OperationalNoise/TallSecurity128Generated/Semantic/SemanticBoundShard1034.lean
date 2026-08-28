import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard986
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1033

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound155374
def owner : Owner := ⟨.program ⟨257⟩, ⟨52487⟩⟩
def transferEvent : Nat := 155374
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 155369 .summary) (.transfer 155373) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 155369 .summary)
      LeftBound155368.bound (LeftBound155368.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50472⟩⟩) (rawTerms := some (Proof.Events606.exact155369RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound155368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 155373)
      LeftBound155373.bound (LeftBound155373.actual selector witness) := by
  exact .transfer (LeftBound155373.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound155368.bound LeftBound155373.bound
def bound : CoeffClass := .finite ⟨2997687391345233100800, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound155368.bound, LeftBound155373.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound155368.actual selector witness) * (LeftBound155373.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound155374

namespace LeftBound155385
def owner : Owner := ⟨.program ⟨257⟩, ⟨51421⟩⟩
def transferEvent : Nat := 155385
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 155383 .coefficient) (.value (.predecessor 1 155384 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 155383 .coefficient)
      LeftAuthority155381.bound (LeftAuthority155381.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events606.exact155382RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority155381.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority155381.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 155384 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority155381.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority155381.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority155381.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound155385

namespace LeftBound155389
def owner : Owner := ⟨.program ⟨257⟩, ⟨51422⟩⟩
def transferEvent : Nat := 155389
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 155387 .coefficient) (.predecessor 1 155388 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 155387 .coefficient)
      LeftBound149117.bound (LeftBound149117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events582.exact149120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound149117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound149117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 155388 .coefficient)
      LeftBound155385.bound (LeftBound155385.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events606.exact155386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound155385.bound, RecordedBoundRefines] <;> decide)
      (LeftBound155385.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound149117.bound LeftBound155385.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound149117.bound, LeftBound155385.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound149117.actual selector witness) * (LeftBound155385.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound155389

namespace LeftBound155390
def owner : Owner := ⟨.program ⟨257⟩, ⟨51422⟩⟩
def transferEvent : Nat := 155390
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨51419⟩⟩]⟩ [⟨.result 155382 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 155382 .coefficient)
      LeftAuthority155381.bound (LeftAuthority155381.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨51419⟩⟩) (rawTerms := some (Proof.Events606.exact155382RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority155381.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority155381.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority155381.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority155381.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority155381.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound155390

namespace LeftBound155391
def owner : Owner := ⟨.program ⟨257⟩, ⟨51422⟩⟩
def transferEvent : Nat := 155391
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 149120 .summary) (.transfer 155390) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 149120 .summary)
      LeftBound149118.bound (LeftBound149118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5545⟩⟩) (rawTerms := some (Proof.Events582.exact149120RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound149118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 155390)
      LeftBound155390.bound (LeftBound155390.actual selector witness) := by
  exact .transfer (LeftBound155390.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound149118.bound LeftBound155390.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound149118.bound, LeftBound155390.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound149118.actual selector witness) * (LeftBound155390.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound155391

namespace LeftBound155470
def owner : Owner := ⟨.program ⟨257⟩, ⟨50465⟩⟩
def transferEvent : Nat := 155470
def frameStart : Nat := 155441
def rule : BoundRule := .product (.predecessor 0 155468 .coefficient) (.predecessor 1 155469 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 155468 .coefficient)
      LeftAuthority155466.bound (LeftAuthority155466.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events607.exact155467RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority155466.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority155466.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 155469 .coefficient)
      LeftAuthority155463.bound (LeftAuthority155463.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events607.exact155464RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority155463.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority155463.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority155466.bound LeftAuthority155463.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority155466.bound, LeftAuthority155463.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority155466.actual selector witness) * (LeftAuthority155463.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound155470

namespace LeftBound155474
def owner : Owner := ⟨.program ⟨257⟩, ⟨50466⟩⟩
def transferEvent : Nat := 155474
def frameStart : Nat := 155441
def rule : BoundRule := .identity (.predecessor 0 155473 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 155473 .coefficient)
      LeftBound155470.bound (LeftBound155470.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events607.exact155472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound155470.bound, RecordedBoundRefines] <;> decide)
      (LeftBound155470.derived selector witness)

def rawBound : CoeffClass := LeftBound155470.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound155470.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound155470.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound155474

namespace LeftBound155491
def owner : Owner := ⟨.program ⟨257⟩, ⟨52274⟩⟩
def transferEvent : Nat := 155491
def frameStart : Nat := 155441
def rule : BoundRule := .sum [.predecessor 0 155489 .coefficient, .predecessor 1 155490 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 155489 .coefficient)
      LeftBound155474.bound (LeftBound155474.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound155474.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 155490 .coefficient)
      LeftAuthority155487.bound (LeftAuthority155487.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority155487.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound155474.bound, LeftAuthority155487.bound]
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound155474.bound, LeftAuthority155487.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound155474.actual selector witness, LeftAuthority155487.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound155491

namespace LeftBound155494
def owner : Owner := ⟨.program ⟨257⟩, ⟨52275⟩⟩
def transferEvent : Nat := 155494
def frameStart : Nat := 155441
def rule : BoundRule := .identity (.predecessor 0 155493 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 155493 .coefficient)
      LeftBound155491.bound (LeftBound155491.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound155491.derived selector witness)

def rawBound : CoeffClass := LeftBound155491.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound155491.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound155491.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound155494

namespace LeftBound155500
def owner : Owner := ⟨.program ⟨257⟩, ⟨52276⟩⟩
def transferEvent : Nat := 155500
def frameStart : Nat := 155441
def rule : BoundRule := .product (.predecessor 0 155498 .coefficient) (.predecessor 1 155499 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 155498 .coefficient)
      LeftAuthority155496.bound (LeftAuthority155496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events607.exact155497RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority155496.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority155496.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 155499 .coefficient)
      LeftBound155494.bound (LeftBound155494.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events607.exact155495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound155494.bound, RecordedBoundRefines] <;> decide)
      (LeftBound155494.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority155496.bound LeftBound155494.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority155496.bound, LeftBound155494.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority155496.actual selector witness) * (LeftBound155494.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound155500

namespace LeftBound155516
def owner : Owner := ⟨.program ⟨257⟩, ⟨9581⟩⟩
def transferEvent : Nat := 155516
def frameStart : Nat := 155441
def rule : BoundRule := .scale (.predecessor 0 155514 .coefficient) (.value (.predecessor 1 155515 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 155514 .coefficient)
      LeftAuthority155512.bound (LeftAuthority155512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events607.exact155513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority155512.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority155512.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 155515 .coefficient)
      LeftAuthority155503.bound (LeftAuthority155503.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority155503.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority155512.bound LeftAuthority155503.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority155512.bound, LeftAuthority155503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority155512.actual selector witness) * (LeftAuthority155503.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound155516

namespace LeftBound155519
def owner : Owner := ⟨.program ⟨257⟩, ⟨7288⟩⟩
def transferEvent : Nat := 155519
def frameStart : Nat := 155441
def rule : BoundRule := .identity (.predecessor 0 155518 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 155518 .coefficient)
      LeftAuthority155506.bound (LeftAuthority155506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events607.exact155507RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority155506.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority155506.derived selector witness)

def rawBound : CoeffClass := LeftAuthority155506.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority155506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority155506.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound155519

namespace LeftBound155523
def owner : Owner := ⟨.program ⟨257⟩, ⟨9582⟩⟩
def transferEvent : Nat := 155523
def frameStart : Nat := 155441
def rule : BoundRule := .product (.predecessor 0 155521 .coefficient) (.predecessor 1 155522 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 155521 .coefficient)
      LeftBound155519.bound (LeftBound155519.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events607.exact155520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound155519.bound, RecordedBoundRefines] <;> decide)
      (LeftBound155519.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 155522 .coefficient)
      LeftBound155516.bound (LeftBound155516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events607.exact155517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound155516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound155516.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound155519.bound LeftBound155516.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound155519.bound, LeftBound155516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound155519.actual selector witness) * (LeftBound155516.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound155523

namespace LeftBound155528
def owner : Owner := ⟨.program ⟨257⟩, ⟨52277⟩⟩
def transferEvent : Nat := 155528
def frameStart : Nat := 155441
def rule : BoundRule := .sum [.predecessor 0 155526 .coefficient, .predecessor 1 155527 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 155526 .coefficient)
      LeftBound155523.bound (LeftBound155523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events607.exact155525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound155523.bound, RecordedBoundRefines] <;> decide)
      (LeftBound155523.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 155527 .coefficient)
      LeftBound155500.bound (LeftBound155500.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events607.exact155502RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound155500.bound, RecordedBoundRefines] <;> decide)
      (LeftBound155500.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound155523.bound, LeftBound155500.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound155523.bound, LeftBound155500.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound155523.actual selector witness, LeftBound155500.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound155528

namespace LeftBound155532
def owner : Owner := ⟨.program ⟨257⟩, ⟨52489⟩⟩
def transferEvent : Nat := 155532
def frameStart : Nat := 155441
def rule : BoundRule := .product (.predecessor 0 155530 .coefficient) (.predecessor 1 155531 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 155530 .coefficient)
      LeftBound155528.bound (LeftBound155528.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events607.exact155529RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound155528.bound, RecordedBoundRefines] <;> decide)
      (LeftBound155528.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 155531 .coefficient)
      LeftAuthority155485.bound (LeftAuthority155485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events607.exact155486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority155485.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority155485.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound155528.bound LeftAuthority155485.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound155528.bound, LeftAuthority155485.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound155528.actual selector witness) * (LeftAuthority155485.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound155532

namespace LeftBound155543
def owner : Owner := ⟨.program ⟨257⟩, ⟨50866⟩⟩
def transferEvent : Nat := 155543
def frameStart : Nat := 155441
def rule : BoundRule := .product (.predecessor 0 155541 .coefficient) (.predecessor 1 155542 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 155541 .coefficient)
      LeftAuthority155496.bound (LeftAuthority155496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events607.exact155497RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority155496.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority155496.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 155542 .coefficient)
      LeftAuthority155539.bound (LeftAuthority155539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events607.exact155540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority155539.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority155539.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority155496.bound LeftAuthority155539.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority155496.bound, LeftAuthority155539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority155496.actual selector witness) * (LeftAuthority155539.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound155543

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
