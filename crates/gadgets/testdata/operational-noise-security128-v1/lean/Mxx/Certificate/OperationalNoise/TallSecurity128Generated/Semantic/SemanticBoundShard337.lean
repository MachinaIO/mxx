import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard133
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard134
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard272
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard275
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard336

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound54821
def owner : Owner := ⟨.program ⟨257⟩, ⟨20905⟩⟩
def transferEvent : Nat := 54821
def frameStart : Nat := 54721
def rule : BoundRule := .sum [.predecessor 0 54819 .coefficient, .predecessor 1 54820 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54819 .coefficient)
      LeftBound54817.bound (LeftBound54817.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54818RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54817.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54817.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54820 .coefficient)
      LeftBound54798.bound (LeftBound54798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54803RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54798.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54798.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54817.bound, LeftBound54798.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54817.bound, LeftBound54798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound54817.actual selector witness, LeftBound54798.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54821

namespace LeftBound54834
def owner : Owner := ⟨.program ⟨257⟩, ⟨20903⟩⟩
def transferEvent : Nat := 54834
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54832 .coefficient, .predecessor 1 54833 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54832 .coefficient)
      LeftBound54663.bound (LeftBound54663.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54831RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54663.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54663.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54833 .coefficient)
      LeftBound54646.bound (LeftBound54646.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54646.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54646.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54663.bound, LeftBound54646.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54663.bound, LeftBound54646.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound54663.actual selector witness, LeftBound54646.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54834

namespace LeftBound54837
def owner : Owner := ⟨.program ⟨257⟩, ⟨20903⟩⟩
def transferEvent : Nat := 54837
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 54831 .summary, .result 54653 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 54831 .summary)
      LeftBound54665.bound (LeftBound54665.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨19619⟩⟩) (rawTerms := some (Proof.Events214.exact54831RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54665.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 54653 .summary)
      LeftBound54648.bound (LeftBound54648.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20902⟩⟩) (rawTerms := some (Proof.Events213.exact54653RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54648.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54665.bound, LeftBound54648.bound]
def bound : CoeffClass := .finite ⟨32188905437706550578131070353408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54665.bound, LeftBound54648.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound54665.actual selector witness, LeftBound54648.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54837

namespace LeftBound54861
def owner : Owner := ⟨.program ⟨257⟩, ⟨15669⟩⟩
def transferEvent : Nat := 54861
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 54859 .coefficient) (.predecessor 1 54860 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54859 .coefficient)
      LeftAuthority1980.bound (LeftAuthority1980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events007.exact1981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1980.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1980.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54860 .coefficient)
      LeftBound46651.bound (LeftBound46651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46651.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46651.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority1980.bound LeftBound46651.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1980.bound, LeftBound46651.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority1980.actual selector witness) * (LeftBound46651.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound54861

namespace LeftBound54866
def owner : Owner := ⟨.program ⟨257⟩, ⟨11210⟩⟩
def transferEvent : Nat := 54866
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 54864 .coefficient) (.predecessor 1 54865 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54864 .coefficient)
      LeftBound46522.bound (LeftBound46522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46522.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46522.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54865 .coefficient)
      LeftBound25596.bound (LeftBound25596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25597RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25596.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25596.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound46522.bound LeftBound25596.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46522.bound, LeftBound25596.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound46522.actual selector witness) * (LeftBound25596.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54866

namespace LeftBound54871
def owner : Owner := ⟨.program ⟨257⟩, ⟨15670⟩⟩
def transferEvent : Nat := 54871
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54869 .coefficient, .predecessor 1 54870 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54869 .coefficient)
      LeftBound54866.bound (LeftBound54866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54866.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54866.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54870 .coefficient)
      LeftBound54861.bound (LeftBound54861.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54863RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54861.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54861.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54866.bound, LeftBound54861.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54866.bound, LeftBound54861.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound54866.actual selector witness, LeftBound54861.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54871

namespace LeftBound54875
def owner : Owner := ⟨.program ⟨257⟩, ⟨15671⟩⟩
def transferEvent : Nat := 54875
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54873 .coefficient, .predecessor 1 54874 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54873 .coefficient)
      LeftBound54871.bound (LeftBound54871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54871.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54874 .coefficient)
      LeftBound25588.bound (LeftBound25588.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25588.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25588.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54871.bound, LeftBound25588.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54871.bound, LeftBound25588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound54871.actual selector witness, LeftBound25588.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54875

namespace LeftBound54876
def owner : Owner := ⟨.program ⟨257⟩, ⟨15671⟩⟩
def transferEvent : Nat := 54876
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨130⟩⟩]⟩ [⟨.result 25589 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 25589 .coefficient)
      LeftBound25588.bound (LeftBound25588.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨130⟩⟩) (rawTerms := some (Proof.Events099.exact25589RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25588.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25588.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound25588.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound25588.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound54876

namespace LeftBound54881
def owner : Owner := ⟨.program ⟨257⟩, ⟨15672⟩⟩
def transferEvent : Nat := 54881
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 54879 .coefficient) (.predecessor 1 54880 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54879 .coefficient)
      LeftBound54875.bound (LeftBound54875.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54875.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54875.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54880 .coefficient)
      LeftAuthority1983.bound (LeftAuthority1983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events007.exact1984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1983.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1983.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound54875.bound LeftAuthority1983.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54875.bound, LeftAuthority1983.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound54875.actual selector witness) * (LeftAuthority1983.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54881

namespace LeftBound54882
def owner : Owner := ⟨.program ⟨257⟩, ⟨15672⟩⟩
def transferEvent : Nat := 54882
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨12501⟩⟩], []⟩ [⟨.result 1984 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 1984 .coefficient)
      LeftAuthority1983.bound (LeftAuthority1983.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨12501⟩⟩) (rawTerms := some (Proof.Events007.exact1984RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1983.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1983.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1983.bound []
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1983.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority1983.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound54882

namespace LeftBound54883
def owner : Owner := ⟨.program ⟨257⟩, ⟨15672⟩⟩
def transferEvent : Nat := 54883
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 54878 .summary) (.transfer 54882) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 54878 .summary)
      LeftBound54876.bound (LeftBound54876.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨15671⟩⟩) (rawTerms := some (Proof.Events214.exact54878RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54876.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 54882)
      LeftBound54882.bound (LeftBound54882.actual selector witness) := by
  exact .transfer (LeftBound54882.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound54876.bound LeftBound54882.bound
def bound : CoeffClass := .finite ⟨1703936, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54876.bound, LeftBound54882.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound54876.actual selector witness) * (LeftBound54882.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54883

namespace LeftBound54889
def owner : Owner := ⟨.program ⟨257⟩, ⟨12502⟩⟩
def transferEvent : Nat := 54889
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 54887 .coefficient) (.predecessor 1 54888 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54887 .coefficient)
      LeftAuthority1983.bound (LeftAuthority1983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events007.exact1984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1983.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1983.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54888 .coefficient)
      LeftBound46651.bound (LeftBound46651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46651.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46651.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority1983.bound LeftBound46651.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1983.bound, LeftBound46651.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority1983.actual selector witness) * (LeftBound46651.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound54889

namespace LeftBound54894
def owner : Owner := ⟨.program ⟨257⟩, ⟨11209⟩⟩
def transferEvent : Nat := 54894
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 54892 .coefficient) (.predecessor 1 54893 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54892 .coefficient)
      LeftBound46522.bound (LeftBound46522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46522.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46522.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54893 .coefficient)
      LeftBound25637.bound (LeftBound25637.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25638RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25637.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25637.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound46522.bound LeftBound25637.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46522.bound, LeftBound25637.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound46522.actual selector witness) * (LeftBound25637.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54894

namespace LeftBound54899
def owner : Owner := ⟨.program ⟨257⟩, ⟨12503⟩⟩
def transferEvent : Nat := 54899
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54897 .coefficient, .predecessor 1 54898 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54897 .coefficient)
      LeftBound54894.bound (LeftBound54894.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54894.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54894.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54898 .coefficient)
      LeftBound54889.bound (LeftBound54889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54889.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54889.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54894.bound, LeftBound54889.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54894.bound, LeftBound54889.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound54894.actual selector witness, LeftBound54889.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54899

namespace LeftBound54903
def owner : Owner := ⟨.program ⟨257⟩, ⟨12504⟩⟩
def transferEvent : Nat := 54903
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54901 .coefficient, .predecessor 1 54902 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54901 .coefficient)
      LeftBound54899.bound (LeftBound54899.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54899.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54899.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54902 .coefficient)
      LeftBound25629.bound (LeftBound25629.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25630RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25629.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25629.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54899.bound, LeftBound25629.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54899.bound, LeftBound25629.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound54899.actual selector witness, LeftBound25629.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54903

namespace LeftBound54904
def owner : Owner := ⟨.program ⟨257⟩, ⟨12504⟩⟩
def transferEvent : Nat := 54904
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨129⟩⟩]⟩ [⟨.result 25630 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 25630 .coefficient)
      LeftBound25629.bound (LeftBound25629.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨129⟩⟩) (rawTerms := some (Proof.Events100.exact25630RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25629.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25629.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound25629.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25629.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound25629.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound54904

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
