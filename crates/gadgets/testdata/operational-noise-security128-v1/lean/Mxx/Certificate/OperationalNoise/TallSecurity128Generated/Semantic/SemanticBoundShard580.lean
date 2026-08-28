import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard067
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard576
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard579

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound90551
def owner : Owner := ⟨.program ⟨257⟩, ⟨47960⟩⟩
def transferEvent : Nat := 90551
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 90549 .coefficient) (.predecessor 1 90550 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90549 .coefficient)
      LeftBound90545.bound (LeftBound90545.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90545.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90545.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90550 .coefficient)
      LeftAuthority3836.bound (LeftAuthority3836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events014.exact3837RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3836.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3836.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound90545.bound LeftAuthority3836.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90545.bound, LeftAuthority3836.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound90545.actual selector witness) * (LeftAuthority3836.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90551

namespace LeftBound90552
def owner : Owner := ⟨.program ⟨257⟩, ⟨47960⟩⟩
def transferEvent : Nat := 90552
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨15156⟩⟩], []⟩ [⟨.result 3837 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 3837 .coefficient)
      LeftAuthority3836.bound (LeftAuthority3836.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨15156⟩⟩) (rawTerms := some (Proof.Events014.exact3837RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3836.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3836.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3836.bound []
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3836.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority3836.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound90552

namespace LeftBound90553
def owner : Owner := ⟨.program ⟨257⟩, ⟨47960⟩⟩
def transferEvent : Nat := 90553
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 90548 .summary) (.transfer 90552) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90548 .summary)
      LeftBound90546.bound (LeftBound90546.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47959⟩⟩) (rawTerms := some (Proof.Events353.exact90548RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90546.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 90552)
      LeftBound90552.bound (LeftBound90552.actual selector witness) := by
  exact .transfer (LeftBound90552.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound90546.bound LeftBound90552.bound
def bound : CoeffClass := .finite ⟨51118080, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90546.bound, LeftBound90552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound90546.actual selector witness) * (LeftBound90552.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90553

namespace LeftBound90559
def owner : Owner := ⟨.program ⟨257⟩, ⟨15157⟩⟩
def transferEvent : Nat := 90559
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 90557 .coefficient) (.predecessor 1 90558 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90557 .coefficient)
      LeftAuthority3836.bound (LeftAuthority3836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events014.exact3837RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3836.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3836.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90558 .coefficient)
      LeftBound90526.bound (LeftBound90526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90526.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority3836.bound LeftBound90526.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3836.bound, LeftBound90526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority3836.actual selector witness) * (LeftBound90526.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound90559

namespace LeftBound90564
def owner : Owner := ⟨.program ⟨257⟩, ⟨9936⟩⟩
def transferEvent : Nat := 90564
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 90562 .coefficient) (.predecessor 1 90563 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90562 .coefficient)
      LeftBound90397.bound (LeftBound90397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90563 .coefficient)
      LeftBound17105.bound (LeftBound17105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events066.exact17106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17105.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17105.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound90397.bound LeftBound17105.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90397.bound, LeftBound17105.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound90397.actual selector witness) * (LeftBound17105.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90564

namespace LeftBound90569
def owner : Owner := ⟨.program ⟨257⟩, ⟨15158⟩⟩
def transferEvent : Nat := 90569
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 90567 .coefficient, .predecessor 1 90568 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90567 .coefficient)
      LeftBound90564.bound (LeftBound90564.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90566RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90564.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90564.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90568 .coefficient)
      LeftBound90559.bound (LeftBound90559.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90559.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90559.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90564.bound, LeftBound90559.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90564.bound, LeftBound90559.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90564.actual selector witness, LeftBound90559.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90569

namespace LeftBound90573
def owner : Owner := ⟨.program ⟨257⟩, ⟨15159⟩⟩
def transferEvent : Nat := 90573
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 90571 .coefficient, .predecessor 1 90572 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90571 .coefficient)
      LeftBound90569.bound (LeftBound90569.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90569.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90569.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90572 .coefficient)
      LeftBound17097.bound (LeftBound17097.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events066.exact17098RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17097.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17097.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90569.bound, LeftBound17097.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90569.bound, LeftBound17097.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90569.actual selector witness, LeftBound17097.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90573

namespace LeftBound90574
def owner : Owner := ⟨.program ⟨257⟩, ⟨15159⟩⟩
def transferEvent : Nat := 90574
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨128⟩⟩]⟩ [⟨.result 17098 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17098 .coefficient)
      LeftBound17097.bound (LeftBound17097.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨128⟩⟩) (rawTerms := some (Proof.Events066.exact17098RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17097.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17097.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound17097.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17097.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound17097.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound90574

namespace LeftBound90579
def owner : Owner := ⟨.program ⟨257⟩, ⟨15160⟩⟩
def transferEvent : Nat := 90579
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 90577 .coefficient) (.predecessor 1 90578 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90577 .coefficient)
      LeftBound90573.bound (LeftBound90573.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90573.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90573.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90578 .coefficient)
      LeftBound17094.bound (LeftBound17094.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events066.exact17095RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17094.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17094.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound90573.bound LeftBound17094.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90573.bound, LeftBound17094.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound90573.actual selector witness) * (LeftBound17094.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90579

namespace LeftBound90580
def owner : Owner := ⟨.program ⟨257⟩, ⟨15160⟩⟩
def transferEvent : Nat := 90580
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩ [⟨.result 17091 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17091 .coefficient)
      LeftAuthority17090.bound (LeftAuthority17090.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9565⟩⟩) (rawTerms := some (Proof.Events066.exact17091RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17090.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17090.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority17090.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17090.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority17090.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound90580

namespace LeftBound90581
def owner : Owner := ⟨.program ⟨257⟩, ⟨15160⟩⟩
def transferEvent : Nat := 90581
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 90576 .summary) (.transfer 90580) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90576 .summary)
      LeftBound90574.bound (LeftBound90574.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨15159⟩⟩) (rawTerms := some (Proof.Events353.exact90576RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90574.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 90580)
      LeftBound90580.bound (LeftBound90580.actual selector witness) := by
  exact .transfer (LeftBound90580.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound90574.bound LeftBound90580.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90574.bound, LeftBound90580.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound90574.actual selector witness) * (LeftBound90580.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90581

namespace LeftBound90589
def owner : Owner := ⟨.program ⟨257⟩, ⟨47961⟩⟩
def transferEvent : Nat := 90589
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 90587 .coefficient, .predecessor 1 90588 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90587 .coefficient)
      LeftBound90579.bound (LeftBound90579.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90579.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90579.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90588 .coefficient)
      LeftBound90551.bound (LeftBound90551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90556RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90551.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90551.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90579.bound, LeftBound90551.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90579.bound, LeftBound90551.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90579.actual selector witness, LeftBound90551.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90589

namespace LeftBound90591
def owner : Owner := ⟨.program ⟨257⟩, ⟨47961⟩⟩
def transferEvent : Nat := 90591
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 90586 .summary, .result 90556 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90586 .summary)
      LeftBound90581.bound (LeftBound90581.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨15160⟩⟩) (rawTerms := some (Proof.Events353.exact90586RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90581.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90556 .summary)
      LeftBound90553.bound (LeftBound90553.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47960⟩⟩) (rawTerms := some (Proof.Events353.exact90556RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90553.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90581.bound, LeftBound90553.bound]
def bound : CoeffClass := .finite ⟨279223992320, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90581.bound, LeftBound90553.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound90581.actual selector witness, LeftBound90553.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90591

namespace LeftBound90595
def owner : Owner := ⟨.program ⟨257⟩, ⟨49715⟩⟩
def transferEvent : Nat := 90595
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 90593 .coefficient) (.predecessor 1 90594 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 90593 .coefficient)
      LeftBound90589.bound (LeftBound90589.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90589.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90589.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 90594 .coefficient)
      LeftAuthority90522.bound (LeftAuthority90522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90522.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90522.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound90589.bound LeftAuthority90522.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90589.bound, LeftAuthority90522.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound90589.actual selector witness) * (LeftAuthority90522.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90595

namespace LeftBound90596
def owner : Owner := ⟨.program ⟨257⟩, ⟨49715⟩⟩
def transferEvent : Nat := 90596
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨49714⟩⟩]⟩ [⟨.result 90523 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90523 .coefficient)
      LeftAuthority90522.bound (LeftAuthority90522.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨49714⟩⟩) (rawTerms := some (Proof.Events353.exact90523RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90522.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90522.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority90522.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90522.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority90522.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound90596

namespace LeftBound90597
def owner : Owner := ⟨.program ⟨257⟩, ⟨49715⟩⟩
def transferEvent : Nat := 90597
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 90592 .summary) (.transfer 90596) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90592 .summary)
      LeftBound90591.bound (LeftBound90591.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47961⟩⟩) (rawTerms := some (Proof.Events353.exact90592RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90591.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 90596)
      LeftBound90596.bound (LeftBound90596.actual selector witness) := by
  exact .transfer (LeftBound90596.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound90591.bound LeftBound90596.bound
def bound : CoeffClass := .finite ⟨2998144788182387916800, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90591.bound, LeftBound90596.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound90591.actual selector witness) * (LeftBound90596.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90597

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
