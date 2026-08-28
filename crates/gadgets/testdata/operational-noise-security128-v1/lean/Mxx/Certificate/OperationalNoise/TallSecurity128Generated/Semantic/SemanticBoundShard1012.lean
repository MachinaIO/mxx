import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard986
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1011

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound152474
def owner : Owner := ⟨.program ⟨257⟩, ⟨26029⟩⟩
def transferEvent : Nat := 152474
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 152472 .coefficient, .predecessor 1 152473 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152472 .coefficient)
      LeftBound152464.bound (LeftBound152464.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152471RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152464.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152464.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152473 .coefficient)
      LeftBound152436.bound (LeftBound152436.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152436.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152436.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound152464.bound, LeftBound152436.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152464.bound, LeftBound152436.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound152464.actual selector witness, LeftBound152436.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound152474

namespace LeftBound152476
def owner : Owner := ⟨.program ⟨257⟩, ⟨26029⟩⟩
def transferEvent : Nat := 152476
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 152471 .summary, .result 152441 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 152471 .summary)
      LeftBound152466.bound (LeftBound152466.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨12940⟩⟩) (rawTerms := some (Proof.Events595.exact152471RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound152466.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 152441 .summary)
      LeftBound152438.bound (LeftBound152438.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨26028⟩⟩) (rawTerms := some (Proof.Events595.exact152441RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound152438.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound152466.bound, LeftBound152438.bound]
def bound : CoeffClass := .finite ⟨279198433280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152466.bound, LeftBound152438.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound152466.actual selector witness, LeftBound152438.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound152476

namespace LeftBound152480
def owner : Owner := ⟨.program ⟨257⟩, ⟨27887⟩⟩
def transferEvent : Nat := 152480
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 152478 .coefficient) (.predecessor 1 152479 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152478 .coefficient)
      LeftBound152474.bound (LeftBound152474.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152477RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152474.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152474.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152479 .coefficient)
      LeftAuthority152412.bound (LeftAuthority152412.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152413RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152412.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152412.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound152474.bound LeftAuthority152412.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152474.bound, LeftAuthority152412.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound152474.actual selector witness) * (LeftAuthority152412.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound152480

namespace LeftBound152481
def owner : Owner := ⟨.program ⟨257⟩, ⟨27887⟩⟩
def transferEvent : Nat := 152481
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨27886⟩⟩]⟩ [⟨.result 152413 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 152413 .coefficient)
      LeftAuthority152412.bound (LeftAuthority152412.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨27886⟩⟩) (rawTerms := some (Proof.Events595.exact152413RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152412.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152412.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority152412.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority152412.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority152412.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound152481

namespace LeftBound152482
def owner : Owner := ⟨.program ⟨257⟩, ⟨27887⟩⟩
def transferEvent : Nat := 152482
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 152477 .summary) (.transfer 152481) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 152477 .summary)
      LeftBound152476.bound (LeftBound152476.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨26029⟩⟩) (rawTerms := some (Proof.Events595.exact152477RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound152476.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 152481)
      LeftBound152481.bound (LeftBound152481.actual selector witness) := by
  exact .transfer (LeftBound152481.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound152476.bound LeftBound152481.bound
def bound : CoeffClass := .finite ⟨2997870350080095027200, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152476.bound, LeftBound152481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound152476.actual selector witness) * (LeftBound152481.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound152482

namespace LeftBound152493
def owner : Owner := ⟨.program ⟨257⟩, ⟨26821⟩⟩
def transferEvent : Nat := 152493
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 152491 .coefficient) (.value (.predecessor 1 152492 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152491 .coefficient)
      LeftAuthority152489.bound (LeftAuthority152489.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152489.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152489.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152492 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority152489.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority152489.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority152489.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound152493

namespace LeftBound152497
def owner : Owner := ⟨.program ⟨257⟩, ⟨26822⟩⟩
def transferEvent : Nat := 152497
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 152495 .coefficient) (.predecessor 1 152496 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152495 .coefficient)
      LeftBound149117.bound (LeftBound149117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events582.exact149120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound149117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound149117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152496 .coefficient)
      LeftBound152493.bound (LeftBound152493.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152493.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152493.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound149117.bound LeftBound152493.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound149117.bound, LeftBound152493.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound149117.actual selector witness) * (LeftBound152493.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound152497

namespace LeftBound152498
def owner : Owner := ⟨.program ⟨257⟩, ⟨26822⟩⟩
def transferEvent : Nat := 152498
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨26819⟩⟩]⟩ [⟨.result 152490 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 152490 .coefficient)
      LeftAuthority152489.bound (LeftAuthority152489.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨26819⟩⟩) (rawTerms := some (Proof.Events595.exact152490RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152489.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152489.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority152489.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority152489.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority152489.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound152498

namespace LeftBound152499
def owner : Owner := ⟨.program ⟨257⟩, ⟨26822⟩⟩
def transferEvent : Nat := 152499
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 149120 .summary) (.transfer 152498) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 149120 .summary)
      LeftBound149118.bound (LeftBound149118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5545⟩⟩) (rawTerms := some (Proof.Events582.exact149120RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound149118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 152498)
      LeftBound152498.bound (LeftBound152498.actual selector witness) := by
  exact .transfer (LeftBound152498.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound149118.bound LeftBound152498.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound149118.bound, LeftBound152498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound149118.actual selector witness) * (LeftBound152498.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound152499

namespace LeftBound152578
def owner : Owner := ⟨.program ⟨257⟩, ⟨26023⟩⟩
def transferEvent : Nat := 152578
def frameStart : Nat := 152549
def rule : BoundRule := .product (.predecessor 0 152576 .coefficient) (.predecessor 1 152577 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152576 .coefficient)
      LeftAuthority152574.bound (LeftAuthority152574.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152575RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152574.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152574.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152577 .coefficient)
      LeftAuthority152571.bound (LeftAuthority152571.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events595.exact152572RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152571.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152571.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority152574.bound LeftAuthority152571.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority152574.bound, LeftAuthority152571.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority152574.actual selector witness) * (LeftAuthority152571.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound152578

namespace LeftBound152582
def owner : Owner := ⟨.program ⟨257⟩, ⟨26024⟩⟩
def transferEvent : Nat := 152582
def frameStart : Nat := 152549
def rule : BoundRule := .identity (.predecessor 0 152581 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152581 .coefficient)
      LeftBound152578.bound (LeftBound152578.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events596.exact152580RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152578.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152578.derived selector witness)

def rawBound : CoeffClass := LeftBound152578.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152578.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound152578.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound152582

namespace LeftBound152599
def owner : Owner := ⟨.program ⟨257⟩, ⟨27674⟩⟩
def transferEvent : Nat := 152599
def frameStart : Nat := 152549
def rule : BoundRule := .sum [.predecessor 0 152597 .coefficient, .predecessor 1 152598 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152597 .coefficient)
      LeftBound152582.bound (LeftBound152582.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound152582.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152598 .coefficient)
      LeftAuthority152595.bound (LeftAuthority152595.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority152595.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound152582.bound, LeftAuthority152595.bound]
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152582.bound, LeftAuthority152595.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound152582.actual selector witness, LeftAuthority152595.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound152599

namespace LeftBound152602
def owner : Owner := ⟨.program ⟨257⟩, ⟨27675⟩⟩
def transferEvent : Nat := 152602
def frameStart : Nat := 152549
def rule : BoundRule := .identity (.predecessor 0 152601 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152601 .coefficient)
      LeftBound152599.bound (LeftBound152599.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound152599.derived selector witness)

def rawBound : CoeffClass := LeftBound152599.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound152599.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound152599.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound152602

namespace LeftBound152608
def owner : Owner := ⟨.program ⟨257⟩, ⟨27676⟩⟩
def transferEvent : Nat := 152608
def frameStart : Nat := 152549
def rule : BoundRule := .product (.predecessor 0 152606 .coefficient) (.predecessor 1 152607 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152606 .coefficient)
      LeftAuthority152604.bound (LeftAuthority152604.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events596.exact152605RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152604.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152604.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152607 .coefficient)
      LeftBound152602.bound (LeftBound152602.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events596.exact152603RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound152602.bound, RecordedBoundRefines] <;> decide)
      (LeftBound152602.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority152604.bound LeftBound152602.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority152604.bound, LeftBound152602.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority152604.actual selector witness) * (LeftBound152602.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound152608

namespace LeftBound152624
def owner : Owner := ⟨.program ⟨257⟩, ⟨9545⟩⟩
def transferEvent : Nat := 152624
def frameStart : Nat := 152549
def rule : BoundRule := .scale (.predecessor 0 152622 .coefficient) (.value (.predecessor 1 152623 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152622 .coefficient)
      LeftAuthority152620.bound (LeftAuthority152620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events596.exact152621RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152620.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152620.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 152623 .coefficient)
      LeftAuthority152611.bound (LeftAuthority152611.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority152611.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority152620.bound LeftAuthority152611.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority152620.bound, LeftAuthority152611.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority152620.actual selector witness) * (LeftAuthority152611.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound152624

namespace LeftBound152627
def owner : Owner := ⟨.program ⟨257⟩, ⟨7295⟩⟩
def transferEvent : Nat := 152627
def frameStart : Nat := 152549
def rule : BoundRule := .identity (.predecessor 0 152626 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 152626 .coefficient)
      LeftAuthority152614.bound (LeftAuthority152614.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events596.exact152615RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority152614.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority152614.derived selector witness)

def rawBound : CoeffClass := LeftAuthority152614.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority152614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority152614.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound152627

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
