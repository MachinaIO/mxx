import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard082
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1286
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1289
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1303

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound194778
def owner : Owner := ⟨.program ⟨257⟩, ⟨41477⟩⟩
def transferEvent : Nat := 194778
def frameStart : Nat := 194705
def rule : BoundRule := .sum [.predecessor 0 194776 .coefficient, .predecessor 1 194777 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 194776 .coefficient)
      LeftAuthority194774.bound (LeftAuthority194774.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events760.exact194775RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority194774.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority194774.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 194777 .coefficient)
      LeftBound194770.bound (LeftBound194770.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events760.exact194772RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound194770.bound, RecordedBoundRefines] <;> decide)
      (LeftBound194770.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority194774.bound, LeftBound194770.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority194774.bound, LeftBound194770.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority194774.actual selector witness, LeftBound194770.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound194778

namespace LeftBound194782
def owner : Owner := ⟨.program ⟨257⟩, ⟨42040⟩⟩
def transferEvent : Nat := 194782
def frameStart : Nat := 194705
def rule : BoundRule := .product (.predecessor 0 194780 .coefficient) (.predecessor 1 194781 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 194780 .coefficient)
      LeftBound194778.bound (LeftBound194778.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events760.exact194779RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound194778.bound, RecordedBoundRefines] <;> decide)
      (LeftBound194778.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 194781 .coefficient)
      LeftAuthority194755.bound (LeftAuthority194755.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events760.exact194756RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority194755.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority194755.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound194778.bound LeftAuthority194755.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound194778.bound, LeftAuthority194755.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound194778.actual selector witness) * (LeftAuthority194755.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound194782

namespace LeftBound194793
def owner : Owner := ⟨.program ⟨257⟩, ⟨40346⟩⟩
def transferEvent : Nat := 194793
def frameStart : Nat := 194705
def rule : BoundRule := .product (.predecessor 0 194791 .coefficient) (.predecessor 1 194792 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 194791 .coefficient)
      LeftAuthority194766.bound (LeftAuthority194766.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events760.exact194767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority194766.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority194766.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 194792 .coefficient)
      LeftAuthority194789.bound (LeftAuthority194789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events760.exact194790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority194789.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority194789.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority194766.bound LeftAuthority194789.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority194766.bound, LeftAuthority194789.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority194766.actual selector witness) * (LeftAuthority194789.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound194793

namespace LeftBound194801
def owner : Owner := ⟨.program ⟨257⟩, ⟨40347⟩⟩
def transferEvent : Nat := 194801
def frameStart : Nat := 194705
def rule : BoundRule := .sum [.predecessor 0 194799 .coefficient, .predecessor 1 194800 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 194799 .coefficient)
      LeftAuthority194797.bound (LeftAuthority194797.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events760.exact194798RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority194797.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority194797.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 194800 .coefficient)
      LeftBound194793.bound (LeftBound194793.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events760.exact194795RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound194793.bound, RecordedBoundRefines] <;> decide)
      (LeftBound194793.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority194797.bound, LeftBound194793.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority194797.bound, LeftBound194793.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority194797.actual selector witness, LeftBound194793.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound194801

namespace LeftBound194805
def owner : Owner := ⟨.program ⟨257⟩, ⟨42043⟩⟩
def transferEvent : Nat := 194805
def frameStart : Nat := 194705
def rule : BoundRule := .sum [.predecessor 0 194803 .coefficient, .predecessor 1 194804 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 194803 .coefficient)
      LeftBound194801.bound (LeftBound194801.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events760.exact194802RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound194801.bound, RecordedBoundRefines] <;> decide)
      (LeftBound194801.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 194804 .coefficient)
      LeftBound194782.bound (LeftBound194782.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events760.exact194787RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound194782.bound, RecordedBoundRefines] <;> decide)
      (LeftBound194782.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound194801.bound, LeftBound194782.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound194801.bound, LeftBound194782.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound194801.actual selector witness, LeftBound194782.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound194805

namespace LeftBound194818
def owner : Owner := ⟨.program ⟨257⟩, ⟨42042⟩⟩
def transferEvent : Nat := 194818
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 194816 .coefficient, .predecessor 1 194817 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 194816 .coefficient)
      LeftBound194647.bound (LeftBound194647.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events760.exact194815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound194647.bound, RecordedBoundRefines] <;> decide)
      (LeftBound194647.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 194817 .coefficient)
      LeftBound194630.bound (LeftBound194630.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events760.exact194637RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound194630.bound, RecordedBoundRefines] <;> decide)
      (LeftBound194630.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound194647.bound, LeftBound194630.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound194647.bound, LeftBound194630.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound194647.actual selector witness, LeftBound194630.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound194818

namespace LeftBound194821
def owner : Owner := ⟨.program ⟨257⟩, ⟨42042⟩⟩
def transferEvent : Nat := 194821
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 194815 .summary, .result 194637 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 194815 .summary)
      LeftBound194649.bound (LeftBound194649.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨40899⟩⟩) (rawTerms := some (Proof.Events760.exact194815RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound194649.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 194637 .summary)
      LeftBound194632.bound (LeftBound194632.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨42041⟩⟩) (rawTerms := some (Proof.Events760.exact194637RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound194632.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound194649.bound, LeftBound194632.bound]
def bound : CoeffClass := .finite ⟨32193129122288829188810200055808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound194649.bound, LeftBound194632.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound194649.actual selector witness, LeftBound194632.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound194821

namespace LeftBound194845
def owner : Owner := ⟨.program ⟨257⟩, ⟨37165⟩⟩
def transferEvent : Nat := 194845
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 194843 .coefficient) (.predecessor 1 194844 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 194843 .coefficient)
      LeftAuthority9161.bound (LeftAuthority9161.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9162RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9161.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9161.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 194844 .coefficient)
      LeftBound192901.bound (LeftBound192901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events753.exact192903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound192901.bound, RecordedBoundRefines] <;> decide)
      (LeftBound192901.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority9161.bound LeftBound192901.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9161.bound, LeftBound192901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority9161.actual selector witness) * (LeftBound192901.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound194845

namespace LeftBound194850
def owner : Owner := ⟨.program ⟨257⟩, ⟨8815⟩⟩
def transferEvent : Nat := 194850
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 194848 .coefficient) (.predecessor 1 194849 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 194848 .coefficient)
      LeftBound192772.bound (LeftBound192772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events753.exact192773RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound192772.bound, RecordedBoundRefines] <;> decide)
      (LeftBound192772.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 194849 .coefficient)
      LeftBound19083.bound (LeftBound19083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19083.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19083.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound192772.bound LeftBound19083.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192772.bound, LeftBound19083.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound192772.actual selector witness) * (LeftBound19083.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound194850

namespace LeftBound194855
def owner : Owner := ⟨.program ⟨257⟩, ⟨37166⟩⟩
def transferEvent : Nat := 194855
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 194853 .coefficient, .predecessor 1 194854 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 194853 .coefficient)
      LeftBound194850.bound (LeftBound194850.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events761.exact194852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound194850.bound, RecordedBoundRefines] <;> decide)
      (LeftBound194850.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 194854 .coefficient)
      LeftBound194845.bound (LeftBound194845.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events761.exact194847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound194845.bound, RecordedBoundRefines] <;> decide)
      (LeftBound194845.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound194850.bound, LeftBound194845.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound194850.bound, LeftBound194845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound194850.actual selector witness, LeftBound194845.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound194855

namespace LeftBound194859
def owner : Owner := ⟨.program ⟨257⟩, ⟨37167⟩⟩
def transferEvent : Nat := 194859
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 194857 .coefficient, .predecessor 1 194858 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 194857 .coefficient)
      LeftBound194855.bound (LeftBound194855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events761.exact194856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound194855.bound, RecordedBoundRefines] <;> decide)
      (LeftBound194855.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 194858 .coefficient)
      LeftBound19075.bound (LeftBound19075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19075.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19075.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound194855.bound, LeftBound19075.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound194855.bound, LeftBound19075.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound194855.actual selector witness, LeftBound19075.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound194859

namespace LeftBound194860
def owner : Owner := ⟨.program ⟨257⟩, ⟨37167⟩⟩
def transferEvent : Nat := 194860
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨107⟩⟩]⟩ [⟨.result 19076 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 19076 .coefficient)
      LeftBound19075.bound (LeftBound19075.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨107⟩⟩) (rawTerms := some (Proof.Events074.exact19076RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19075.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19075.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound19075.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19075.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound19075.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound194860

namespace LeftBound194865
def owner : Owner := ⟨.program ⟨257⟩, ⟨37168⟩⟩
def transferEvent : Nat := 194865
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 194863 .coefficient) (.predecessor 1 194864 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 194863 .coefficient)
      LeftBound194859.bound (LeftBound194859.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events761.exact194862RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound194859.bound, RecordedBoundRefines] <;> decide)
      (LeftBound194859.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 194864 .coefficient)
      LeftAuthority9164.bound (LeftAuthority9164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9164.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9164.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound194859.bound LeftAuthority9164.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound194859.bound, LeftAuthority9164.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound194859.actual selector witness) * (LeftAuthority9164.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound194865

namespace LeftBound194866
def owner : Owner := ⟨.program ⟨257⟩, ⟨37168⟩⟩
def transferEvent : Nat := 194866
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨13911⟩⟩], []⟩ [⟨.result 9165 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 9165 .coefficient)
      LeftAuthority9164.bound (LeftAuthority9164.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨13911⟩⟩) (rawTerms := some (Proof.Events035.exact9165RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9164.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9164.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority9164.bound []
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9164.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority9164.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound194866

namespace LeftBound194867
def owner : Owner := ⟨.program ⟨257⟩, ⟨37168⟩⟩
def transferEvent : Nat := 194867
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 194862 .summary) (.transfer 194866) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 194862 .summary)
      LeftBound194860.bound (LeftBound194860.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨37167⟩⟩) (rawTerms := some (Proof.Events761.exact194862RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound194860.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 194866)
      LeftBound194866.bound (LeftBound194866.actual selector witness) := by
  exact .transfer (LeftBound194866.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound194860.bound LeftBound194866.bound
def bound : CoeffClass := .finite ⟨35782656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound194860.bound, LeftBound194866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound194860.actual selector witness) * (LeftBound194866.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound194867

namespace LeftBound194873
def owner : Owner := ⟨.program ⟨257⟩, ⟨13912⟩⟩
def transferEvent : Nat := 194873
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 194871 .coefficient) (.predecessor 1 194872 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 194871 .coefficient)
      LeftAuthority9164.bound (LeftAuthority9164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9164.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 194872 .coefficient)
      LeftBound192901.bound (LeftBound192901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events753.exact192903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound192901.bound, RecordedBoundRefines] <;> decide)
      (LeftBound192901.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority9164.bound LeftBound192901.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9164.bound, LeftBound192901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority9164.actual selector witness) * (LeftBound192901.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound194873

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
