import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1494
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1540

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound228493
def owner : Owner := ⟨.program ⟨257⟩, ⟨50526⟩⟩
def transferEvent : Nat := 228493
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 228488 .summary, .result 228458 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 228488 .summary)
      LeftBound228483.bound (LeftBound228483.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50525⟩⟩) (rawTerms := some (Proof.Events892.exact228488RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound228483.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 228458 .summary)
      LeftBound228455.bound (LeftBound228455.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50521⟩⟩) (rawTerms := some (Proof.Events892.exact228458RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound228455.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound228483.bound, LeftBound228455.bound]
def bound : CoeffClass := .finite ⟨279181393920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound228483.bound, LeftBound228455.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound228483.actual selector witness, LeftBound228455.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound228493

namespace LeftBound228497
def owner : Owner := ⟨.program ⟨257⟩, ⟨52509⟩⟩
def transferEvent : Nat := 228497
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 228495 .coefficient) (.predecessor 1 228496 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 228495 .coefficient)
      LeftBound228491.bound (LeftBound228491.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events892.exact228494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound228491.bound, RecordedBoundRefines] <;> decide)
      (LeftBound228491.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 228496 .coefficient)
      LeftAuthority228429.bound (LeftAuthority228429.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events892.exact228430RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority228429.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority228429.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound228491.bound LeftAuthority228429.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound228491.bound, LeftAuthority228429.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound228491.actual selector witness) * (LeftAuthority228429.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound228497

namespace LeftBound228498
def owner : Owner := ⟨.program ⟨257⟩, ⟨52509⟩⟩
def transferEvent : Nat := 228498
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩ [⟨.result 228430 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 228430 .coefficient)
      LeftAuthority228429.bound (LeftAuthority228429.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨52508⟩⟩) (rawTerms := some (Proof.Events892.exact228430RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority228429.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority228429.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority228429.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority228429.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority228429.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound228498

namespace LeftBound228499
def owner : Owner := ⟨.program ⟨257⟩, ⟨52509⟩⟩
def transferEvent : Nat := 228499
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 228494 .summary) (.transfer 228498) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 228494 .summary)
      LeftBound228493.bound (LeftBound228493.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨50526⟩⟩) (rawTerms := some (Proof.Events892.exact228494RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound228493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 228498)
      LeftBound228498.bound (LeftBound228498.actual selector witness) := by
  exact .transfer (LeftBound228498.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound228493.bound LeftBound228498.bound
def bound : CoeffClass := .finite ⟨2997687391345233100800, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound228493.bound, LeftBound228498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound228493.actual selector witness) * (LeftBound228498.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound228499

namespace LeftBound228510
def owner : Owner := ⟨.program ⟨257⟩, ⟨51441⟩⟩
def transferEvent : Nat := 228510
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 228508 .coefficient) (.value (.predecessor 1 228509 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 228508 .coefficient)
      LeftAuthority228506.bound (LeftAuthority228506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events892.exact228507RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority228506.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority228506.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 228509 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority228506.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority228506.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority228506.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound228510

namespace LeftBound228514
def owner : Owner := ⟨.program ⟨257⟩, ⟨51442⟩⟩
def transferEvent : Nat := 228514
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 228512 .coefficient) (.predecessor 1 228513 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 228512 .coefficient)
      LeftBound222242.bound (LeftBound222242.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events868.exact222245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222242.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 228513 .coefficient)
      LeftBound228510.bound (LeftBound228510.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events892.exact228511RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound228510.bound, RecordedBoundRefines] <;> decide)
      (LeftBound228510.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound222242.bound LeftBound228510.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222242.bound, LeftBound228510.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound222242.actual selector witness) * (LeftBound228510.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound228514

namespace LeftBound228515
def owner : Owner := ⟨.program ⟨257⟩, ⟨51442⟩⟩
def transferEvent : Nat := 228515
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨51439⟩⟩]⟩ [⟨.result 228507 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 228507 .coefficient)
      LeftAuthority228506.bound (LeftAuthority228506.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨51439⟩⟩) (rawTerms := some (Proof.Events892.exact228507RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority228506.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority228506.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority228506.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority228506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority228506.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound228515

namespace LeftBound228516
def owner : Owner := ⟨.program ⟨257⟩, ⟨51442⟩⟩
def transferEvent : Nat := 228516
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 222245 .summary) (.transfer 228515) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 222245 .summary)
      LeftBound222243.bound (LeftBound222243.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5581⟩⟩) (rawTerms := some (Proof.Events868.exact222245RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound222243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 228515)
      LeftBound228515.bound (LeftBound228515.actual selector witness) := by
  exact .transfer (LeftBound228515.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound222243.bound LeftBound228515.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222243.bound, LeftBound228515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound222243.actual selector witness) * (LeftBound228515.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound228516

namespace LeftBound228595
def owner : Owner := ⟨.program ⟨257⟩, ⟨50519⟩⟩
def transferEvent : Nat := 228595
def frameStart : Nat := 228566
def rule : BoundRule := .product (.predecessor 0 228593 .coefficient) (.predecessor 1 228594 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 228593 .coefficient)
      LeftAuthority228591.bound (LeftAuthority228591.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events892.exact228592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority228591.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority228591.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 228594 .coefficient)
      LeftAuthority228588.bound (LeftAuthority228588.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events892.exact228589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority228588.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority228588.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority228591.bound LeftAuthority228588.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority228591.bound, LeftAuthority228588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority228591.actual selector witness) * (LeftAuthority228588.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound228595

namespace LeftBound228599
def owner : Owner := ⟨.program ⟨257⟩, ⟨50520⟩⟩
def transferEvent : Nat := 228599
def frameStart : Nat := 228566
def rule : BoundRule := .identity (.predecessor 0 228598 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 228598 .coefficient)
      LeftBound228595.bound (LeftBound228595.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events892.exact228597RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound228595.bound, RecordedBoundRefines] <;> decide)
      (LeftBound228595.derived selector witness)

def rawBound : CoeffClass := LeftBound228595.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound228595.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound228595.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound228599

namespace LeftBound228616
def owner : Owner := ⟨.program ⟨257⟩, ⟨52282⟩⟩
def transferEvent : Nat := 228616
def frameStart : Nat := 228566
def rule : BoundRule := .sum [.predecessor 0 228614 .coefficient, .predecessor 1 228615 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 228614 .coefficient)
      LeftBound228599.bound (LeftBound228599.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound228599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 228615 .coefficient)
      LeftAuthority228612.bound (LeftAuthority228612.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority228612.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound228599.bound, LeftAuthority228612.bound]
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound228599.bound, LeftAuthority228612.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound228599.actual selector witness, LeftAuthority228612.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound228616

namespace LeftBound228619
def owner : Owner := ⟨.program ⟨257⟩, ⟨52283⟩⟩
def transferEvent : Nat := 228619
def frameStart : Nat := 228566
def rule : BoundRule := .identity (.predecessor 0 228618 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 228618 .coefficient)
      LeftBound228616.bound (LeftBound228616.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound228616.derived selector witness)

def rawBound : CoeffClass := LeftBound228616.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound228616.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound228616.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound228619

namespace LeftBound228625
def owner : Owner := ⟨.program ⟨257⟩, ⟨52284⟩⟩
def transferEvent : Nat := 228625
def frameStart : Nat := 228566
def rule : BoundRule := .product (.predecessor 0 228623 .coefficient) (.predecessor 1 228624 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 228623 .coefficient)
      LeftAuthority228621.bound (LeftAuthority228621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events893.exact228622RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority228621.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority228621.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 228624 .coefficient)
      LeftBound228619.bound (LeftBound228619.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events893.exact228620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound228619.bound, RecordedBoundRefines] <;> decide)
      (LeftBound228619.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority228621.bound LeftBound228619.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority228621.bound, LeftBound228619.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority228621.actual selector witness) * (LeftBound228619.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound228625

namespace LeftBound228641
def owner : Owner := ⟨.program ⟨257⟩, ⟨9581⟩⟩
def transferEvent : Nat := 228641
def frameStart : Nat := 228566
def rule : BoundRule := .scale (.predecessor 0 228639 .coefficient) (.value (.predecessor 1 228640 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 228639 .coefficient)
      LeftAuthority228637.bound (LeftAuthority228637.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events893.exact228638RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority228637.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority228637.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 228640 .coefficient)
      LeftAuthority228628.bound (LeftAuthority228628.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority228628.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority228637.bound LeftAuthority228628.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority228637.bound, LeftAuthority228628.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority228637.actual selector witness) * (LeftAuthority228628.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound228641

namespace LeftBound228644
def owner : Owner := ⟨.program ⟨257⟩, ⟨7288⟩⟩
def transferEvent : Nat := 228644
def frameStart : Nat := 228566
def rule : BoundRule := .identity (.predecessor 0 228643 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 228643 .coefficient)
      LeftAuthority228631.bound (LeftAuthority228631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events893.exact228632RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority228631.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority228631.derived selector witness)

def rawBound : CoeffClass := LeftAuthority228631.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority228631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority228631.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound228644

namespace LeftBound228648
def owner : Owner := ⟨.program ⟨257⟩, ⟨9582⟩⟩
def transferEvent : Nat := 228648
def frameStart : Nat := 228566
def rule : BoundRule := .product (.predecessor 0 228646 .coefficient) (.predecessor 1 228647 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 228646 .coefficient)
      LeftBound228644.bound (LeftBound228644.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events893.exact228645RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound228644.bound, RecordedBoundRefines] <;> decide)
      (LeftBound228644.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 228647 .coefficient)
      LeftBound228641.bound (LeftBound228641.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events893.exact228642RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound228641.bound, RecordedBoundRefines] <;> decide)
      (LeftBound228641.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound228644.bound LeftBound228641.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound228644.bound, LeftBound228641.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound228644.actual selector witness) * (LeftBound228641.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound228648

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
