import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard074
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard075
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1591
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1594
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1600

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound237681
def owner : Owner := ⟨.program ⟨257⟩, ⟨46820⟩⟩
def transferEvent : Nat := 237681
def frameStart : Nat := 237616
def rule : BoundRule := .product (.predecessor 0 237679 .coefficient) (.predecessor 1 237680 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237679 .coefficient)
      LeftAuthority237677.bound (LeftAuthority237677.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events928.exact237678RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority237677.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority237677.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 237680 .coefficient)
      LeftBound237675.bound (LeftBound237675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events928.exact237676RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237675.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237675.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority237677.bound LeftBound237675.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority237677.bound, LeftBound237675.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority237677.actual selector witness) * (LeftBound237675.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound237681

namespace LeftBound237689
def owner : Owner := ⟨.program ⟨257⟩, ⟨46821⟩⟩
def transferEvent : Nat := 237689
def frameStart : Nat := 237616
def rule : BoundRule := .sum [.predecessor 0 237687 .coefficient, .predecessor 1 237688 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237687 .coefficient)
      LeftAuthority237685.bound (LeftAuthority237685.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events928.exact237686RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority237685.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority237685.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 237688 .coefficient)
      LeftBound237681.bound (LeftBound237681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events928.exact237683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237681.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237681.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority237685.bound, LeftBound237681.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority237685.bound, LeftBound237681.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority237685.actual selector witness, LeftBound237681.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound237689

namespace LeftBound237693
def owner : Owner := ⟨.program ⟨257⟩, ⟨47300⟩⟩
def transferEvent : Nat := 237693
def frameStart : Nat := 237616
def rule : BoundRule := .product (.predecessor 0 237691 .coefficient) (.predecessor 1 237692 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237691 .coefficient)
      LeftBound237689.bound (LeftBound237689.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events928.exact237690RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237689.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237689.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 237692 .coefficient)
      LeftAuthority237666.bound (LeftAuthority237666.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events928.exact237667RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority237666.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority237666.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound237689.bound LeftAuthority237666.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound237689.bound, LeftAuthority237666.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound237689.actual selector witness) * (LeftAuthority237666.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound237693

namespace LeftBound237704
def owner : Owner := ⟨.program ⟨257⟩, ⟨45658⟩⟩
def transferEvent : Nat := 237704
def frameStart : Nat := 237616
def rule : BoundRule := .product (.predecessor 0 237702 .coefficient) (.predecessor 1 237703 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237702 .coefficient)
      LeftAuthority237677.bound (LeftAuthority237677.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events928.exact237678RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority237677.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority237677.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 237703 .coefficient)
      LeftAuthority237700.bound (LeftAuthority237700.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events928.exact237701RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority237700.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority237700.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority237677.bound LeftAuthority237700.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority237677.bound, LeftAuthority237700.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority237677.actual selector witness) * (LeftAuthority237700.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound237704

namespace LeftBound237712
def owner : Owner := ⟨.program ⟨257⟩, ⟨45659⟩⟩
def transferEvent : Nat := 237712
def frameStart : Nat := 237616
def rule : BoundRule := .sum [.predecessor 0 237710 .coefficient, .predecessor 1 237711 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237710 .coefficient)
      LeftAuthority237708.bound (LeftAuthority237708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events928.exact237709RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority237708.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority237708.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 237711 .coefficient)
      LeftBound237704.bound (LeftBound237704.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events928.exact237706RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237704.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237704.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority237708.bound, LeftBound237704.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority237708.bound, LeftBound237704.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority237708.actual selector witness, LeftBound237704.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound237712

namespace LeftBound237716
def owner : Owner := ⟨.program ⟨257⟩, ⟨47303⟩⟩
def transferEvent : Nat := 237716
def frameStart : Nat := 237616
def rule : BoundRule := .sum [.predecessor 0 237714 .coefficient, .predecessor 1 237715 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237714 .coefficient)
      LeftBound237712.bound (LeftBound237712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events928.exact237713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237712.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 237715 .coefficient)
      LeftBound237693.bound (LeftBound237693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events928.exact237698RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237693.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237693.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound237712.bound, LeftBound237693.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound237712.bound, LeftBound237693.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound237712.actual selector witness, LeftBound237693.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound237716

namespace LeftBound237729
def owner : Owner := ⟨.program ⟨257⟩, ⟨47302⟩⟩
def transferEvent : Nat := 237729
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 237727 .coefficient, .predecessor 1 237728 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237727 .coefficient)
      LeftBound237558.bound (LeftBound237558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events928.exact237726RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 237728 .coefficient)
      LeftBound237541.bound (LeftBound237541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events927.exact237548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237541.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237541.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound237558.bound, LeftBound237541.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound237558.bound, LeftBound237541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound237558.actual selector witness, LeftBound237541.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound237729

namespace LeftBound237732
def owner : Owner := ⟨.program ⟨257⟩, ⟨47302⟩⟩
def transferEvent : Nat := 237732
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 237726 .summary, .result 237548 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 237726 .summary)
      LeftBound237560.bound (LeftBound237560.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨46179⟩⟩) (rawTerms := some (Proof.Events928.exact237726RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound237560.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 237548 .summary)
      LeftBound237543.bound (LeftBound237543.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47301⟩⟩) (rawTerms := some (Proof.Events927.exact237548RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound237543.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound237560.bound, LeftBound237543.bound]
def bound : CoeffClass := .finite ⟨32194307824962953452255538577408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound237560.bound, LeftBound237543.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound237560.actual selector witness, LeftBound237543.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound237732

namespace LeftBound237756
def owner : Owner := ⟨.program ⟨257⟩, ⟨42429⟩⟩
def transferEvent : Nat := 237756
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 237754 .coefficient) (.predecessor 1 237755 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237754 .coefficient)
      LeftAuthority11359.bound (LeftAuthority11359.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11359.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11359.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 237755 .coefficient)
      LeftBound236776.bound (LeftBound236776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events924.exact236778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236776.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority11359.bound LeftBound236776.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11359.bound, LeftBound236776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority11359.actual selector witness) * (LeftBound236776.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound237756

namespace LeftBound237761
def owner : Owner := ⟨.program ⟨257⟩, ⟨8361⟩⟩
def transferEvent : Nat := 237761
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 237759 .coefficient) (.predecessor 1 237760 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237759 .coefficient)
      LeftBound236647.bound (LeftBound236647.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events924.exact236648RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236647.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236647.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 237760 .coefficient)
      LeftBound18081.bound (LeftBound18081.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18081.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18081.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound236647.bound LeftBound18081.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236647.bound, LeftBound18081.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound236647.actual selector witness) * (LeftBound18081.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound237761

namespace LeftBound237766
def owner : Owner := ⟨.program ⟨257⟩, ⟨42430⟩⟩
def transferEvent : Nat := 237766
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 237764 .coefficient, .predecessor 1 237765 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237764 .coefficient)
      LeftBound237761.bound (LeftBound237761.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events928.exact237763RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237761.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237761.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 237765 .coefficient)
      LeftBound237756.bound (LeftBound237756.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events928.exact237758RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237756.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237756.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound237761.bound, LeftBound237756.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound237761.bound, LeftBound237756.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound237761.actual selector witness, LeftBound237756.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound237766

namespace LeftBound237770
def owner : Owner := ⟨.program ⟨257⟩, ⟨42431⟩⟩
def transferEvent : Nat := 237770
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 237768 .coefficient, .predecessor 1 237769 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237768 .coefficient)
      LeftBound237766.bound (LeftBound237766.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events928.exact237767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237766.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237766.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 237769 .coefficient)
      LeftBound18073.bound (LeftBound18073.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18074RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18073.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18073.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound237766.bound, LeftBound18073.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound237766.bound, LeftBound18073.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound237766.actual selector witness, LeftBound18073.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound237770

namespace LeftBound237771
def owner : Owner := ⟨.program ⟨257⟩, ⟨42431⟩⟩
def transferEvent : Nat := 237771
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨109⟩⟩]⟩ [⟨.result 18074 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 18074 .coefficient)
      LeftBound18073.bound (LeftBound18073.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨109⟩⟩) (rawTerms := some (Proof.Events070.exact18074RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18073.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18073.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound18073.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18073.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound18073.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound237771

namespace LeftBound237776
def owner : Owner := ⟨.program ⟨257⟩, ⟨42432⟩⟩
def transferEvent : Nat := 237776
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 237774 .coefficient) (.predecessor 1 237775 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 237774 .coefficient)
      LeftBound237770.bound (LeftBound237770.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events928.exact237773RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237770.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237770.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 237775 .coefficient)
      LeftAuthority11362.bound (LeftAuthority11362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11363RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11362.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11362.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound237770.bound LeftAuthority11362.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound237770.bound, LeftAuthority11362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound237770.actual selector witness) * (LeftAuthority11362.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound237776

namespace LeftBound237777
def owner : Owner := ⟨.program ⟨257⟩, ⟨42432⟩⟩
def transferEvent : Nat := 237777
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨14451⟩⟩], []⟩ [⟨.result 11363 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 11363 .coefficient)
      LeftAuthority11362.bound (LeftAuthority11362.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨14451⟩⟩) (rawTerms := some (Proof.Events044.exact11363RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11362.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11362.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority11362.bound []
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority11362.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound237777

namespace LeftBound237778
def owner : Owner := ⟨.program ⟨257⟩, ⟨42432⟩⟩
def transferEvent : Nat := 237778
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 237773 .summary) (.transfer 237777) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 237773 .summary)
      LeftBound237771.bound (LeftBound237771.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨42431⟩⟩) (rawTerms := some (Proof.Events928.exact237773RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound237771.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 237777)
      LeftBound237777.bound (LeftBound237777.actual selector witness) := by
  exact .transfer (LeftBound237777.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound237771.bound LeftBound237777.bound
def bound : CoeffClass := .finite ⟨44302336, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound237771.bound, LeftBound237777.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound237771.actual selector witness) * (LeftBound237777.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound237778

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
