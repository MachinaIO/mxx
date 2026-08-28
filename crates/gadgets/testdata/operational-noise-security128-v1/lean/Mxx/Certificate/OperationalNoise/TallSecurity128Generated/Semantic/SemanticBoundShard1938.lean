import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard110
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1895
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1898
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1937

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound285880
def owner : Owner := ⟨.program ⟨257⟩, ⟨61285⟩⟩
def transferEvent : Nat := 285880
def frameStart : Nat := 285807
def rule : BoundRule := .sum [.predecessor 0 285878 .coefficient, .predecessor 1 285879 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285878 .coefficient)
      LeftAuthority285876.bound (LeftAuthority285876.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1116.exact285877RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority285876.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority285876.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285879 .coefficient)
      LeftBound285872.bound (LeftBound285872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1116.exact285874RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285872.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority285876.bound, LeftBound285872.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority285876.bound, LeftBound285872.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority285876.actual selector witness, LeftBound285872.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound285880

namespace LeftBound285884
def owner : Owner := ⟨.program ⟨257⟩, ⟨61707⟩⟩
def transferEvent : Nat := 285884
def frameStart : Nat := 285807
def rule : BoundRule := .product (.predecessor 0 285882 .coefficient) (.predecessor 1 285883 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285882 .coefficient)
      LeftBound285880.bound (LeftBound285880.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1116.exact285881RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285880.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285880.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285883 .coefficient)
      LeftAuthority285857.bound (LeftAuthority285857.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1116.exact285858RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority285857.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority285857.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound285880.bound LeftAuthority285857.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound285880.bound, LeftAuthority285857.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound285880.actual selector witness) * (LeftAuthority285857.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound285884

namespace LeftBound285895
def owner : Owner := ⟨.program ⟨257⟩, ⟨59989⟩⟩
def transferEvent : Nat := 285895
def frameStart : Nat := 285807
def rule : BoundRule := .product (.predecessor 0 285893 .coefficient) (.predecessor 1 285894 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285893 .coefficient)
      LeftAuthority285868.bound (LeftAuthority285868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1116.exact285869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority285868.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority285868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285894 .coefficient)
      LeftAuthority285891.bound (LeftAuthority285891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1116.exact285892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority285891.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority285891.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority285868.bound LeftAuthority285891.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority285868.bound, LeftAuthority285891.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority285868.actual selector witness) * (LeftAuthority285891.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound285895

namespace LeftBound285903
def owner : Owner := ⟨.program ⟨257⟩, ⟨59990⟩⟩
def transferEvent : Nat := 285903
def frameStart : Nat := 285807
def rule : BoundRule := .sum [.predecessor 0 285901 .coefficient, .predecessor 1 285902 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285901 .coefficient)
      LeftAuthority285899.bound (LeftAuthority285899.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1116.exact285900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority285899.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority285899.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285902 .coefficient)
      LeftBound285895.bound (LeftBound285895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1116.exact285897RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285895.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285895.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority285899.bound, LeftBound285895.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority285899.bound, LeftBound285895.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority285899.actual selector witness, LeftBound285895.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound285903

namespace LeftBound285907
def owner : Owner := ⟨.program ⟨257⟩, ⟨61711⟩⟩
def transferEvent : Nat := 285907
def frameStart : Nat := 285807
def rule : BoundRule := .sum [.predecessor 0 285905 .coefficient, .predecessor 1 285906 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285905 .coefficient)
      LeftBound285903.bound (LeftBound285903.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1116.exact285904RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285903.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285903.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285906 .coefficient)
      LeftBound285884.bound (LeftBound285884.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1116.exact285889RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285884.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285884.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound285903.bound, LeftBound285884.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound285903.bound, LeftBound285884.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound285903.actual selector witness, LeftBound285884.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound285907

namespace LeftBound285920
def owner : Owner := ⟨.program ⟨257⟩, ⟨61709⟩⟩
def transferEvent : Nat := 285920
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 285918 .coefficient, .predecessor 1 285919 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285918 .coefficient)
      LeftBound285749.bound (LeftBound285749.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1116.exact285917RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285749.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285749.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285919 .coefficient)
      LeftBound285732.bound (LeftBound285732.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1116.exact285739RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285732.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285732.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound285749.bound, LeftBound285732.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound285749.bound, LeftBound285732.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound285749.actual selector witness, LeftBound285732.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound285920

namespace LeftBound285923
def owner : Owner := ⟨.program ⟨257⟩, ⟨61709⟩⟩
def transferEvent : Nat := 285923
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 285917 .summary, .result 285739 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 285917 .summary)
      LeftBound285751.bound (LeftBound285751.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨60579⟩⟩) (rawTerms := some (Proof.Events1116.exact285917RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound285751.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 285739 .summary)
      LeftBound285734.bound (LeftBound285734.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61708⟩⟩) (rawTerms := some (Proof.Events1116.exact285739RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound285734.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound285751.bound, LeftBound285734.bound]
def bound : CoeffClass := .finite ⟨32190378816049205907437743505408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound285751.bound, LeftBound285734.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound285751.actual selector witness, LeftBound285734.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound285923

namespace LeftBound285947
def owner : Owner := ⟨.program ⟨257⟩, ⟨24939⟩⟩
def transferEvent : Nat := 285947
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 285945 .coefficient) (.predecessor 1 285946 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285945 .coefficient)
      LeftAuthority13804.bound (LeftAuthority13804.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13805RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13804.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13804.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285946 .coefficient)
      LeftBound280651.bound (LeftBound280651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1096.exact280653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280651.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280651.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority13804.bound LeftBound280651.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13804.bound, LeftBound280651.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority13804.actual selector witness) * (LeftBound280651.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound285947

namespace LeftBound285952
def owner : Owner := ⟨.program ⟨257⟩, ⟨7895⟩⟩
def transferEvent : Nat := 285952
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 285950 .coefficient) (.predecessor 1 285951 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285950 .coefficient)
      LeftBound280522.bound (LeftBound280522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1095.exact280523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280522.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280522.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285951 .coefficient)
      LeftBound22590.bound (LeftBound22590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22591RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22590.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22590.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound280522.bound LeftBound22590.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280522.bound, LeftBound22590.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound280522.actual selector witness) * (LeftBound22590.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound285952

namespace LeftBound285957
def owner : Owner := ⟨.program ⟨257⟩, ⟨24940⟩⟩
def transferEvent : Nat := 285957
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 285955 .coefficient, .predecessor 1 285956 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285955 .coefficient)
      LeftBound285952.bound (LeftBound285952.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact285954RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285952.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285952.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285956 .coefficient)
      LeftBound285947.bound (LeftBound285947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1116.exact285949RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285947.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285947.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound285952.bound, LeftBound285947.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound285952.bound, LeftBound285947.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound285952.actual selector witness, LeftBound285947.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound285957

namespace LeftBound285961
def owner : Owner := ⟨.program ⟨257⟩, ⟨24941⟩⟩
def transferEvent : Nat := 285961
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 285959 .coefficient, .predecessor 1 285960 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285959 .coefficient)
      LeftBound285957.bound (LeftBound285957.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact285958RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285957.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285957.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285960 .coefficient)
      LeftBound22582.bound (LeftBound22582.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22582.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22582.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound285957.bound, LeftBound22582.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound285957.bound, LeftBound22582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound285957.actual selector witness, LeftBound22582.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound285961

namespace LeftBound285962
def owner : Owner := ⟨.program ⟨257⟩, ⟨24941⟩⟩
def transferEvent : Nat := 285962
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨99⟩⟩]⟩ [⟨.result 22583 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 22583 .coefficient)
      LeftBound22582.bound (LeftBound22582.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨99⟩⟩) (rawTerms := some (Proof.Events088.exact22583RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22582.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22582.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound22582.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound22582.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound285962

namespace LeftBound285967
def owner : Owner := ⟨.program ⟨257⟩, ⟨56346⟩⟩
def transferEvent : Nat := 285967
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 285965 .coefficient) (.predecessor 1 285966 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285965 .coefficient)
      LeftBound285961.bound (LeftBound285961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1117.exact285964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285961.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285961.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285966 .coefficient)
      LeftAuthority13807.bound (LeftAuthority13807.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13808RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13807.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13807.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound285961.bound LeftAuthority13807.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound285961.bound, LeftAuthority13807.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound285961.actual selector witness) * (LeftAuthority13807.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound285967

namespace LeftBound285968
def owner : Owner := ⟨.program ⟨257⟩, ⟨56346⟩⟩
def transferEvent : Nat := 285968
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨56343⟩⟩], []⟩ [⟨.result 13808 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 13808 .coefficient)
      LeftAuthority13807.bound (LeftAuthority13807.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨56343⟩⟩) (rawTerms := some (Proof.Events053.exact13808RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13807.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13807.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority13807.bound []
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13807.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority13807.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound285968

namespace LeftBound285969
def owner : Owner := ⟨.program ⟨257⟩, ⟨56346⟩⟩
def transferEvent : Nat := 285969
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 285964 .summary) (.transfer 285968) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 285964 .summary)
      LeftBound285962.bound (LeftBound285962.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24941⟩⟩) (rawTerms := some (Proof.Events1117.exact285964RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound285962.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 285968)
      LeftBound285968.bound (LeftBound285968.actual selector witness) := by
  exact .transfer (LeftBound285968.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound285962.bound LeftBound285968.bound
def bound : CoeffClass := .finite ⟨13631488, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound285962.bound, LeftBound285968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound285962.actual selector witness) * (LeftBound285968.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound285969

namespace LeftBound285975
def owner : Owner := ⟨.program ⟨257⟩, ⟨56347⟩⟩
def transferEvent : Nat := 285975
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 285973 .coefficient) (.predecessor 1 285974 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 285973 .coefficient)
      LeftAuthority13807.bound (LeftAuthority13807.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13808RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13807.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13807.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 285974 .coefficient)
      LeftBound280651.bound (LeftBound280651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1096.exact280653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280651.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280651.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority13807.bound LeftBound280651.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13807.bound, LeftBound280651.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority13807.actual selector witness) * (LeftBound280651.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound285975

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
