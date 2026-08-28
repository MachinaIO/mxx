import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard094
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard095
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard374
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard376
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard401

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound64603
def owner : Owner := ⟨.program ⟨257⟩, ⟨31145⟩⟩
def transferEvent : Nat := 64603
def frameStart : Nat := 64526
def rule : BoundRule := .product (.predecessor 0 64601 .coefficient) (.predecessor 1 64602 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64601 .coefficient)
      LeftBound64599.bound (LeftBound64599.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64599.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64602 .coefficient)
      LeftAuthority64576.bound (LeftAuthority64576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64577RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64576.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64576.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound64599.bound LeftAuthority64576.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64599.bound, LeftAuthority64576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound64599.actual selector witness) * (LeftAuthority64576.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64603

namespace LeftBound64614
def owner : Owner := ⟨.program ⟨257⟩, ⟨29391⟩⟩
def transferEvent : Nat := 64614
def frameStart : Nat := 64526
def rule : BoundRule := .product (.predecessor 0 64612 .coefficient) (.predecessor 1 64613 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64612 .coefficient)
      LeftAuthority64587.bound (LeftAuthority64587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64587.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64587.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64613 .coefficient)
      LeftAuthority64610.bound (LeftAuthority64610.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64610.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64610.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority64587.bound LeftAuthority64610.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64587.bound, LeftAuthority64610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority64587.actual selector witness) * (LeftAuthority64610.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64614

namespace LeftBound64622
def owner : Owner := ⟨.program ⟨257⟩, ⟨29392⟩⟩
def transferEvent : Nat := 64622
def frameStart : Nat := 64526
def rule : BoundRule := .sum [.predecessor 0 64620 .coefficient, .predecessor 1 64621 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64620 .coefficient)
      LeftAuthority64618.bound (LeftAuthority64618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64618.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64621 .coefficient)
      LeftBound64614.bound (LeftBound64614.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64614.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64614.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority64618.bound, LeftBound64614.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64618.bound, LeftBound64614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority64618.actual selector witness, LeftBound64614.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64622

namespace LeftBound64626
def owner : Owner := ⟨.program ⟨257⟩, ⟨31148⟩⟩
def transferEvent : Nat := 64626
def frameStart : Nat := 64526
def rule : BoundRule := .sum [.predecessor 0 64624 .coefficient, .predecessor 1 64625 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64624 .coefficient)
      LeftBound64622.bound (LeftBound64622.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64623RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64622.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64622.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64625 .coefficient)
      LeftBound64603.bound (LeftBound64603.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64603.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64603.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64622.bound, LeftBound64603.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64622.bound, LeftBound64603.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound64622.actual selector witness, LeftBound64603.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64626

namespace LeftBound64639
def owner : Owner := ⟨.program ⟨257⟩, ⟨31147⟩⟩
def transferEvent : Nat := 64639
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64637 .coefficient, .predecessor 1 64638 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64637 .coefficient)
      LeftBound64468.bound (LeftBound64468.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64468.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64468.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64638 .coefficient)
      LeftBound64451.bound (LeftBound64451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events251.exact64458RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64451.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64451.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64468.bound, LeftBound64451.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64468.bound, LeftBound64451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound64468.actual selector witness, LeftBound64451.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64639

namespace LeftBound64642
def owner : Owner := ⟨.program ⟨257⟩, ⟨31147⟩⟩
def transferEvent : Nat := 64642
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64636 .summary, .result 64458 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 64636 .summary)
      LeftBound64470.bound (LeftBound64470.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨29979⟩⟩) (rawTerms := some (Proof.Events252.exact64636RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64470.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 64458 .summary)
      LeftBound64453.bound (LeftBound64453.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31146⟩⟩) (rawTerms := some (Proof.Events251.exact64458RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64453.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64470.bound, LeftBound64453.bound]
def bound : CoeffClass := .finite ⟨32192146870060392302605751287808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64470.bound, LeftBound64453.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound64470.actual selector witness, LeftBound64453.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64642

namespace LeftBound64666
def owner : Owner := ⟨.program ⟨257⟩, ⟨26265⟩⟩
def transferEvent : Nat := 64666
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 64664 .coefficient) (.predecessor 1 64665 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64664 .coefficient)
      LeftAuthority2498.bound (LeftAuthority2498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2499RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2498.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2498.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64665 .coefficient)
      LeftBound61276.bound (LeftBound61276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61276.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61276.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority2498.bound LeftBound61276.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2498.bound, LeftBound61276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority2498.actual selector witness) * (LeftBound61276.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound64666

namespace LeftBound64671
def owner : Owner := ⟨.program ⟨257⟩, ⟨10760⟩⟩
def transferEvent : Nat := 64671
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 64669 .coefficient) (.predecessor 1 64670 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64669 .coefficient)
      LeftBound61147.bound (LeftBound61147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61147.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61147.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64670 .coefficient)
      LeftBound20586.bound (LeftBound20586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20587RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20586.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound61147.bound LeftBound20586.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61147.bound, LeftBound20586.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound61147.actual selector witness) * (LeftBound20586.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64671

namespace LeftBound64676
def owner : Owner := ⟨.program ⟨257⟩, ⟨26266⟩⟩
def transferEvent : Nat := 64676
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64674 .coefficient, .predecessor 1 64675 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64674 .coefficient)
      LeftBound64671.bound (LeftBound64671.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64671.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64671.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64675 .coefficient)
      LeftBound64666.bound (LeftBound64666.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64668RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64666.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64666.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64671.bound, LeftBound64666.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64671.bound, LeftBound64666.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound64671.actual selector witness, LeftBound64666.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64676

namespace LeftBound64680
def owner : Owner := ⟨.program ⟨257⟩, ⟨26267⟩⟩
def transferEvent : Nat := 64680
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64678 .coefficient, .predecessor 1 64679 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64678 .coefficient)
      LeftBound64676.bound (LeftBound64676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64676.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64676.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64679 .coefficient)
      LeftBound20578.bound (LeftBound20578.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20579RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20578.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20578.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64676.bound, LeftBound20578.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64676.bound, LeftBound20578.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound64676.actual selector witness, LeftBound20578.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64680

namespace LeftBound64681
def owner : Owner := ⟨.program ⟨257⟩, ⟨26267⟩⟩
def transferEvent : Nat := 64681
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨104⟩⟩]⟩ [⟨.result 20579 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 20579 .coefficient)
      LeftBound20578.bound (LeftBound20578.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨104⟩⟩) (rawTerms := some (Proof.Events080.exact20579RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20578.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20578.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound20578.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20578.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound20578.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound64681

namespace LeftBound64686
def owner : Owner := ⟨.program ⟨257⟩, ⟨26268⟩⟩
def transferEvent : Nat := 64686
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 64684 .coefficient) (.predecessor 1 64685 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64684 .coefficient)
      LeftBound64680.bound (LeftBound64680.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64680.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64680.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64685 .coefficient)
      LeftAuthority2501.bound (LeftAuthority2501.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2502RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2501.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2501.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound64680.bound LeftAuthority2501.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64680.bound, LeftAuthority2501.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound64680.actual selector witness) * (LeftAuthority2501.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64686

namespace LeftBound64687
def owner : Owner := ⟨.program ⟨257⟩, ⟨26268⟩⟩
def transferEvent : Nat := 64687
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨13086⟩⟩], []⟩ [⟨.result 2502 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 2502 .coefficient)
      LeftAuthority2501.bound (LeftAuthority2501.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨13086⟩⟩) (rawTerms := some (Proof.Events009.exact2502RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2501.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2501.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority2501.bound []
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2501.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority2501.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound64687

namespace LeftBound64688
def owner : Owner := ⟨.program ⟨257⟩, ⟨26268⟩⟩
def transferEvent : Nat := 64688
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 64683 .summary) (.transfer 64687) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 64683 .summary)
      LeftBound64681.bound (LeftBound64681.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨26267⟩⟩) (rawTerms := some (Proof.Events252.exact64683RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64681.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 64687)
      LeftBound64687.bound (LeftBound64687.actual selector witness) := by
  exact .transfer (LeftBound64687.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound64681.bound LeftBound64687.bound
def bound : CoeffClass := .finite ⟨25559040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64681.bound, LeftBound64687.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound64681.actual selector witness) * (LeftBound64687.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64688

namespace LeftBound64694
def owner : Owner := ⟨.program ⟨257⟩, ⟨13087⟩⟩
def transferEvent : Nat := 64694
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 64692 .coefficient) (.predecessor 1 64693 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64692 .coefficient)
      LeftAuthority2501.bound (LeftAuthority2501.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2502RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2501.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2501.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64693 .coefficient)
      LeftBound61276.bound (LeftBound61276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61276.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61276.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority2501.bound LeftBound61276.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2501.bound, LeftBound61276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority2501.actual selector witness) * (LeftBound61276.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound64694

namespace LeftBound64699
def owner : Owner := ⟨.program ⟨257⟩, ⟨10777⟩⟩
def transferEvent : Nat := 64699
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 64697 .coefficient) (.predecessor 1 64698 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64697 .coefficient)
      LeftBound61147.bound (LeftBound61147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61147.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61147.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64698 .coefficient)
      LeftBound20627.bound (LeftBound20627.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20627.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20627.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound61147.bound LeftBound20627.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61147.bound, LeftBound20627.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound61147.actual selector witness) * (LeftBound20627.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64699

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
