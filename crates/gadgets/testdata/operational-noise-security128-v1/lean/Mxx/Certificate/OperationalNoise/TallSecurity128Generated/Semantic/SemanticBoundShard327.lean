import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard122
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard276
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard326

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound53457
def owner : Owner := ⟨.program ⟨257⟩, ⟨31707⟩⟩
def transferEvent : Nat := 53457
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 53455 .coefficient, .predecessor 1 53456 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 53455 .coefficient)
      LeftBound53453.bound (LeftBound53453.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53454RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53453.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53453.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 53456 .coefficient)
      LeftBound24126.bound (LeftBound24126.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24127RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24126.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24126.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53453.bound, LeftBound24126.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53453.bound, LeftBound24126.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound53453.actual selector witness, LeftBound24126.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53457

namespace LeftBound53458
def owner : Owner := ⟨.program ⟨257⟩, ⟨31707⟩⟩
def transferEvent : Nat := 53458
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨113⟩⟩]⟩ [⟨.result 24127 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 24127 .coefficient)
      LeftBound24126.bound (LeftBound24126.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨113⟩⟩) (rawTerms := some (Proof.Events094.exact24127RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24126.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24126.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound24126.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24126.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound24126.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound53458

namespace LeftBound53463
def owner : Owner := ⟨.program ⟨257⟩, ⟨31708⟩⟩
def transferEvent : Nat := 53463
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 53461 .coefficient) (.predecessor 1 53462 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 53461 .coefficient)
      LeftBound53457.bound (LeftBound53457.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53457.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53457.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 53462 .coefficient)
      LeftBound24123.bound (LeftBound24123.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24124RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24123.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24123.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound53457.bound LeftBound24123.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53457.bound, LeftBound24123.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound53457.actual selector witness) * (LeftBound24123.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53463

namespace LeftBound53464
def owner : Owner := ⟨.program ⟨257⟩, ⟨31708⟩⟩
def transferEvent : Nat := 53464
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩ [⟨.result 24120 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 24120 .coefficient)
      LeftAuthority24119.bound (LeftAuthority24119.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9577⟩⟩) (rawTerms := some (Proof.Events094.exact24120RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24119.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24119.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority24119.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24119.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority24119.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound53464

namespace LeftBound53465
def owner : Owner := ⟨.program ⟨257⟩, ⟨31708⟩⟩
def transferEvent : Nat := 53465
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 53460 .summary) (.transfer 53464) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 53460 .summary)
      LeftBound53458.bound (LeftBound53458.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31707⟩⟩) (rawTerms := some (Proof.Events208.exact53460RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53458.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 53464)
      LeftBound53464.bound (LeftBound53464.actual selector witness) := by
  exact .transfer (LeftBound53464.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound53458.bound LeftBound53464.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53458.bound, LeftBound53464.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound53458.actual selector witness) * (LeftBound53464.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53465

namespace LeftBound53473
def owner : Owner := ⟨.program ⟨257⟩, ⟨31709⟩⟩
def transferEvent : Nat := 53473
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 53471 .coefficient, .predecessor 1 53472 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 53471 .coefficient)
      LeftBound53463.bound (LeftBound53463.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53470RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53463.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53463.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 53472 .coefficient)
      LeftBound53435.bound (LeftBound53435.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53440RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53435.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53435.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53463.bound, LeftBound53435.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53463.bound, LeftBound53435.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound53463.actual selector witness, LeftBound53435.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53473

namespace LeftBound53475
def owner : Owner := ⟨.program ⟨257⟩, ⟨31709⟩⟩
def transferEvent : Nat := 53475
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 53470 .summary, .result 53440 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 53470 .summary)
      LeftBound53465.bound (LeftBound53465.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31708⟩⟩) (rawTerms := some (Proof.Events208.exact53470RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53465.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 53440 .summary)
      LeftBound53437.bound (LeftBound53437.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31704⟩⟩) (rawTerms := some (Proof.Events208.exact53440RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53437.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53465.bound, LeftBound53437.bound]
def bound : CoeffClass := .finite ⟨279177986048, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53465.bound, LeftBound53437.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound53465.actual selector witness, LeftBound53437.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53475

namespace LeftBound53479
def owner : Owner := ⟨.program ⟨257⟩, ⟨33548⟩⟩
def transferEvent : Nat := 53479
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 53477 .coefficient) (.predecessor 1 53478 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 53477 .coefficient)
      LeftBound53473.bound (LeftBound53473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53473.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 53478 .coefficient)
      LeftAuthority53411.bound (LeftAuthority53411.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53412RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53411.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53411.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound53473.bound LeftAuthority53411.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53473.bound, LeftAuthority53411.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound53473.actual selector witness) * (LeftAuthority53411.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53479

namespace LeftBound53480
def owner : Owner := ⟨.program ⟨257⟩, ⟨33548⟩⟩
def transferEvent : Nat := 53480
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨33547⟩⟩]⟩ [⟨.result 53412 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 53412 .coefficient)
      LeftAuthority53411.bound (LeftAuthority53411.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨33547⟩⟩) (rawTerms := some (Proof.Events208.exact53412RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53411.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53411.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority53411.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53411.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority53411.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound53480

namespace LeftBound53481
def owner : Owner := ⟨.program ⟨257⟩, ⟨33548⟩⟩
def transferEvent : Nat := 53481
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 53476 .summary) (.transfer 53480) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 53476 .summary)
      LeftBound53475.bound (LeftBound53475.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31709⟩⟩) (rawTerms := some (Proof.Events208.exact53476RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53475.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 53480)
      LeftBound53480.bound (LeftBound53480.actual selector witness) := by
  exact .transfer (LeftBound53480.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound53475.bound LeftBound53480.bound
def bound : CoeffClass := .finite ⟨2997650799598260715520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53475.bound, LeftBound53480.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound53475.actual selector witness) * (LeftBound53480.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53481

namespace LeftBound53492
def owner : Owner := ⟨.program ⟨257⟩, ⟨32471⟩⟩
def transferEvent : Nat := 53492
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 53490 .coefficient) (.value (.predecessor 1 53491 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 53490 .coefficient)
      LeftAuthority53488.bound (LeftAuthority53488.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53489RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53488.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53488.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 53491 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority53488.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53488.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority53488.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound53492

namespace LeftBound53496
def owner : Owner := ⟨.program ⟨257⟩, ⟨32472⟩⟩
def transferEvent : Nat := 53496
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 53494 .coefficient) (.predecessor 1 53495 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 53494 .coefficient)
      LeftBound46742.bound (LeftBound46742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 53495 .coefficient)
      LeftBound53492.bound (LeftBound53492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53493RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53492.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53492.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound46742.bound LeftBound53492.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46742.bound, LeftBound53492.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound46742.actual selector witness) * (LeftBound53492.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53496

namespace LeftBound53497
def owner : Owner := ⟨.program ⟨257⟩, ⟨32472⟩⟩
def transferEvent : Nat := 53497
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨32469⟩⟩]⟩ [⟨.result 53489 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 53489 .coefficient)
      LeftAuthority53488.bound (LeftAuthority53488.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨32469⟩⟩) (rawTerms := some (Proof.Events208.exact53489RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53488.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53488.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority53488.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53488.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority53488.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound53497

namespace LeftBound53498
def owner : Owner := ⟨.program ⟨257⟩, ⟨32472⟩⟩
def transferEvent : Nat := 53498
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 46745 .summary) (.transfer 53497) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46745 .summary)
      LeftBound46743.bound (LeftBound46743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨11216⟩⟩) (rawTerms := some (Proof.Events182.exact46745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 53497)
      LeftBound53497.bound (LeftBound53497.actual selector witness) := by
  exact .transfer (LeftBound53497.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound46743.bound LeftBound53497.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46743.bound, LeftBound53497.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound46743.actual selector witness) * (LeftBound53497.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53498

namespace LeftBound53577
def owner : Owner := ⟨.program ⟨257⟩, ⟨31702⟩⟩
def transferEvent : Nat := 53577
def frameStart : Nat := 53548
def rule : BoundRule := .product (.predecessor 0 53575 .coefficient) (.predecessor 1 53576 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 53575 .coefficient)
      LeftAuthority53573.bound (LeftAuthority53573.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53574RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53573.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53573.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 53576 .coefficient)
      LeftAuthority53570.bound (LeftAuthority53570.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53571RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53570.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53570.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority53573.bound LeftAuthority53570.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53573.bound, LeftAuthority53570.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority53573.actual selector witness) * (LeftAuthority53570.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53577

namespace LeftBound53581
def owner : Owner := ⟨.program ⟨257⟩, ⟨31703⟩⟩
def transferEvent : Nat := 53581
def frameStart : Nat := 53548
def rule : BoundRule := .identity (.predecessor 0 53580 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 53580 .coefficient)
      LeftBound53577.bound (LeftBound53577.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53579RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53577.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53577.derived selector witness)

def rawBound : CoeffClass := LeftBound53577.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53577.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound53577.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound53581

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
