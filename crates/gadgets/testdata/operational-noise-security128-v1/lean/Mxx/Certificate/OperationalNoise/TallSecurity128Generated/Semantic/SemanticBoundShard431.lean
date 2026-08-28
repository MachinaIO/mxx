import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard126
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard374
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard376
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard430

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound68459
def owner : Owner := ⟨.program ⟨257⟩, ⟨34110⟩⟩
def transferEvent : Nat := 68459
def frameStart : Nat := 68382
def rule : BoundRule := .product (.predecessor 0 68457 .coefficient) (.predecessor 1 68458 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 68457 .coefficient)
      LeftBound68455.bound (LeftBound68455.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68456RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68455.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68455.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 68458 .coefficient)
      LeftAuthority68432.bound (LeftAuthority68432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68432.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68432.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound68455.bound LeftAuthority68432.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68455.bound, LeftAuthority68432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound68455.actual selector witness) * (LeftAuthority68432.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68459

namespace LeftBound68470
def owner : Owner := ⟨.program ⟨257⟩, ⟨32241⟩⟩
def transferEvent : Nat := 68470
def frameStart : Nat := 68382
def rule : BoundRule := .product (.predecessor 0 68468 .coefficient) (.predecessor 1 68469 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 68468 .coefficient)
      LeftAuthority68443.bound (LeftAuthority68443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68444RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68443.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68443.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 68469 .coefficient)
      LeftAuthority68466.bound (LeftAuthority68466.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68467RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68466.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68466.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority68443.bound LeftAuthority68466.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68443.bound, LeftAuthority68466.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority68443.actual selector witness) * (LeftAuthority68466.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68470

namespace LeftBound68478
def owner : Owner := ⟨.program ⟨257⟩, ⟨32242⟩⟩
def transferEvent : Nat := 68478
def frameStart : Nat := 68382
def rule : BoundRule := .sum [.predecessor 0 68476 .coefficient, .predecessor 1 68477 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 68476 .coefficient)
      LeftAuthority68474.bound (LeftAuthority68474.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68474.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68474.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 68477 .coefficient)
      LeftBound68470.bound (LeftBound68470.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68470.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68470.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority68474.bound, LeftBound68470.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68474.bound, LeftBound68470.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority68474.actual selector witness, LeftBound68470.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68478

namespace LeftBound68482
def owner : Owner := ⟨.program ⟨257⟩, ⟨34114⟩⟩
def transferEvent : Nat := 68482
def frameStart : Nat := 68382
def rule : BoundRule := .sum [.predecessor 0 68480 .coefficient, .predecessor 1 68481 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 68480 .coefficient)
      LeftBound68478.bound (LeftBound68478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68479RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68478.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68478.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 68481 .coefficient)
      LeftBound68459.bound (LeftBound68459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68464RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68459.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68478.bound, LeftBound68459.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68478.bound, LeftBound68459.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound68478.actual selector witness, LeftBound68459.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68482

namespace LeftBound68495
def owner : Owner := ⟨.program ⟨257⟩, ⟨34112⟩⟩
def transferEvent : Nat := 68495
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 68493 .coefficient, .predecessor 1 68494 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 68493 .coefficient)
      LeftBound68324.bound (LeftBound68324.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68492RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68324.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68324.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 68494 .coefficient)
      LeftBound68307.bound (LeftBound68307.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68307.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68307.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68324.bound, LeftBound68307.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68324.bound, LeftBound68307.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound68324.actual selector witness, LeftBound68307.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68495

namespace LeftBound68498
def owner : Owner := ⟨.program ⟨257⟩, ⟨34112⟩⟩
def transferEvent : Nat := 68498
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 68492 .summary, .result 68314 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 68492 .summary)
      LeftBound68326.bound (LeftBound68326.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨32839⟩⟩) (rawTerms := some (Proof.Events267.exact68492RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68326.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 68314 .summary)
      LeftBound68309.bound (LeftBound68309.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34111⟩⟩) (rawTerms := some (Proof.Events266.exact68314RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68309.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68326.bound, LeftBound68309.bound]
def bound : CoeffClass := .finite ⟨32189200113375081643992404983808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68326.bound, LeftBound68309.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound68326.actual selector witness, LeftBound68309.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68498

namespace LeftBound68522
def owner : Owner := ⟨.program ⟨257⟩, ⟨21665⟩⟩
def transferEvent : Nat := 68522
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 68520 .coefficient) (.predecessor 1 68521 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 68520 .coefficient)
      LeftAuthority2682.bound (LeftAuthority2682.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events010.exact2683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2682.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 68521 .coefficient)
      LeftBound61276.bound (LeftBound61276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61276.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61276.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority2682.bound LeftBound61276.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2682.bound, LeftBound61276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority2682.actual selector witness) * (LeftBound61276.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound68522

namespace LeftBound68527
def owner : Owner := ⟨.program ⟨257⟩, ⟨10788⟩⟩
def transferEvent : Nat := 68527
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 68525 .coefficient) (.predecessor 1 68526 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 68525 .coefficient)
      LeftBound61147.bound (LeftBound61147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61147.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61147.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 68526 .coefficient)
      LeftBound24594.bound (LeftBound24594.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24595RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24594.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24594.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound61147.bound LeftBound24594.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61147.bound, LeftBound24594.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound61147.actual selector witness) * (LeftBound24594.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68527

namespace LeftBound68532
def owner : Owner := ⟨.program ⟨257⟩, ⟨21666⟩⟩
def transferEvent : Nat := 68532
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 68530 .coefficient, .predecessor 1 68531 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 68530 .coefficient)
      LeftBound68527.bound (LeftBound68527.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68529RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68527.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68527.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 68531 .coefficient)
      LeftBound68522.bound (LeftBound68522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68524RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68522.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68522.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68527.bound, LeftBound68522.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68527.bound, LeftBound68522.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound68527.actual selector witness, LeftBound68522.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68532

namespace LeftBound68536
def owner : Owner := ⟨.program ⟨257⟩, ⟨21667⟩⟩
def transferEvent : Nat := 68536
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 68534 .coefficient, .predecessor 1 68535 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 68534 .coefficient)
      LeftBound68532.bound (LeftBound68532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68533RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68532.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68532.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 68535 .coefficient)
      LeftBound24586.bound (LeftBound24586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24587RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24586.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68532.bound, LeftBound24586.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68532.bound, LeftBound24586.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound68532.actual selector witness, LeftBound24586.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68536

namespace LeftBound68537
def owner : Owner := ⟨.program ⟨257⟩, ⟨21667⟩⟩
def transferEvent : Nat := 68537
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨132⟩⟩]⟩ [⟨.result 24587 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 24587 .coefficient)
      LeftBound24586.bound (LeftBound24586.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨132⟩⟩) (rawTerms := some (Proof.Events096.exact24587RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24586.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound24586.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24586.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound24586.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound68537

namespace LeftBound68542
def owner : Owner := ⟨.program ⟨257⟩, ⟨21668⟩⟩
def transferEvent : Nat := 68542
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 68540 .coefficient) (.predecessor 1 68541 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 68540 .coefficient)
      LeftBound68536.bound (LeftBound68536.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68539RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68536.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68536.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 68541 .coefficient)
      LeftAuthority2685.bound (LeftAuthority2685.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events010.exact2686RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2685.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2685.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound68536.bound LeftAuthority2685.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68536.bound, LeftAuthority2685.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound68536.actual selector witness) * (LeftAuthority2685.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68542

namespace LeftBound68543
def owner : Owner := ⟨.program ⟨257⟩, ⟨21668⟩⟩
def transferEvent : Nat := 68543
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨21206⟩⟩], []⟩ [⟨.result 2686 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 2686 .coefficient)
      LeftAuthority2685.bound (LeftAuthority2685.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨21206⟩⟩) (rawTerms := some (Proof.Events010.exact2686RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2685.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2685.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority2685.bound []
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2685.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority2685.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound68543

namespace LeftBound68544
def owner : Owner := ⟨.program ⟨257⟩, ⟨21668⟩⟩
def transferEvent : Nat := 68544
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 68539 .summary) (.transfer 68543) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 68539 .summary)
      LeftBound68537.bound (LeftBound68537.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨21667⟩⟩) (rawTerms := some (Proof.Events267.exact68539RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68537.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 68543)
      LeftBound68543.bound (LeftBound68543.actual selector witness) := by
  exact .transfer (LeftBound68543.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound68537.bound LeftBound68543.bound
def bound : CoeffClass := .finite ⟨3407872, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68537.bound, LeftBound68543.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound68537.actual selector witness) * (LeftBound68543.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68544

namespace LeftBound68550
def owner : Owner := ⟨.program ⟨257⟩, ⟨21207⟩⟩
def transferEvent : Nat := 68550
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 68548 .coefficient) (.predecessor 1 68549 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 68548 .coefficient)
      LeftAuthority2685.bound (LeftAuthority2685.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events010.exact2686RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2685.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2685.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 68549 .coefficient)
      LeftBound61276.bound (LeftBound61276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61276.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61276.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority2685.bound LeftBound61276.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2685.bound, LeftBound61276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority2685.actual selector witness) * (LeftBound61276.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound68550

namespace LeftBound68555
def owner : Owner := ⟨.program ⟨257⟩, ⟨10768⟩⟩
def transferEvent : Nat := 68555
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 68553 .coefficient) (.predecessor 1 68554 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 68553 .coefficient)
      LeftBound61147.bound (LeftBound61147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61147.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61147.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 68554 .coefficient)
      LeftBound24635.bound (LeftBound24635.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24635.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24635.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound61147.bound LeftBound24635.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61147.bound, LeftBound24635.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound61147.actual selector witness) * (LeftBound24635.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68555

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
