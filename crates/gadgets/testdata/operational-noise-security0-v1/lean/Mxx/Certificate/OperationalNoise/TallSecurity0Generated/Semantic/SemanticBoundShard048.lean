import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard028
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard047

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound8894
def owner : Owner := ⟨.program ⟨214⟩, ⟨16607⟩⟩
def transferEvent : Nat := 8894
def frameStart : Nat := 8829
def rule : BoundRule := .product (.predecessor 0 8892 .coefficient) (.predecessor 1 8893 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8892 .coefficient)
      LeftAuthority8890.bound (LeftAuthority8890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8890.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8890.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8893 .coefficient)
      LeftBound8888.bound (LeftBound8888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8889RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8888.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8888.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority8890.bound LeftBound8888.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8890.bound, LeftBound8888.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority8890.actual selector witness) * (LeftBound8888.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8894

namespace LeftBound8902
def owner : Owner := ⟨.program ⟨214⟩, ⟨16608⟩⟩
def transferEvent : Nat := 8902
def frameStart : Nat := 8829
def rule : BoundRule := .sum [.predecessor 0 8900 .coefficient, .predecessor 1 8901 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8900 .coefficient)
      LeftAuthority8898.bound (LeftAuthority8898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8898.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8898.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8901 .coefficient)
      LeftBound8894.bound (LeftBound8894.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8894.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8894.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority8898.bound, LeftBound8894.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8898.bound, LeftBound8894.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority8898.actual selector witness, LeftBound8894.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8902

namespace LeftBound8906
def owner : Owner := ⟨.program ⟨214⟩, ⟨29221⟩⟩
def transferEvent : Nat := 8906
def frameStart : Nat := 8829
def rule : BoundRule := .product (.predecessor 0 8904 .coefficient) (.predecessor 1 8905 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8904 .coefficient)
      LeftBound8902.bound (LeftBound8902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8902.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8902.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8905 .coefficient)
      LeftAuthority8879.bound (LeftAuthority8879.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8880RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8879.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8879.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound8902.bound LeftAuthority8879.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8902.bound, LeftAuthority8879.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound8902.actual selector witness) * (LeftAuthority8879.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8906

namespace LeftBound8917
def owner : Owner := ⟨.program ⟨214⟩, ⟨18218⟩⟩
def transferEvent : Nat := 8917
def frameStart : Nat := 8829
def rule : BoundRule := .product (.predecessor 0 8915 .coefficient) (.predecessor 1 8916 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8915 .coefficient)
      LeftAuthority8890.bound (LeftAuthority8890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8890.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8890.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8916 .coefficient)
      LeftAuthority8913.bound (LeftAuthority8913.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8913.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8913.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority8890.bound LeftAuthority8913.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8890.bound, LeftAuthority8913.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority8890.actual selector witness) * (LeftAuthority8913.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8917

namespace LeftBound8925
def owner : Owner := ⟨.program ⟨214⟩, ⟨18219⟩⟩
def transferEvent : Nat := 8925
def frameStart : Nat := 8829
def rule : BoundRule := .sum [.predecessor 0 8923 .coefficient, .predecessor 1 8924 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8923 .coefficient)
      LeftAuthority8921.bound (LeftAuthority8921.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8922RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8921.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8921.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8924 .coefficient)
      LeftBound8917.bound (LeftBound8917.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8919RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8917.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8917.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority8921.bound, LeftBound8917.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8921.bound, LeftBound8917.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority8921.actual selector witness, LeftBound8917.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8925

namespace LeftBound8929
def owner : Owner := ⟨.program ⟨214⟩, ⟨29225⟩⟩
def transferEvent : Nat := 8929
def frameStart : Nat := 8829
def rule : BoundRule := .sum [.predecessor 0 8927 .coefficient, .predecessor 1 8928 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8927 .coefficient)
      LeftBound8925.bound (LeftBound8925.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8926RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8925.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8925.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8928 .coefficient)
      LeftBound8906.bound (LeftBound8906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8906.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8906.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8925.bound, LeftBound8906.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8925.bound, LeftBound8906.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound8925.actual selector witness, LeftBound8906.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8929

namespace LeftBound8942
def owner : Owner := ⟨.program ⟨214⟩, ⟨29223⟩⟩
def transferEvent : Nat := 8942
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8940 .coefficient, .predecessor 1 8941 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8940 .coefficient)
      LeftBound8771.bound (LeftBound8771.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8939RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8771.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8771.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8941 .coefficient)
      LeftBound8754.bound (LeftBound8754.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8761RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8754.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8754.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8771.bound, LeftBound8754.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8771.bound, LeftBound8754.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound8771.actual selector witness, LeftBound8754.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8942

namespace LeftBound8945
def owner : Owner := ⟨.program ⟨214⟩, ⟨29223⟩⟩
def transferEvent : Nat := 8945
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 8939 .summary, .result 8761 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8939 .summary)
      LeftBound8773.bound (LeftBound8773.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22283⟩⟩) (rawTerms := some (Proof.Events034.exact8939RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound8773.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8761 .summary)
      LeftBound8756.bound (LeftBound8756.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29222⟩⟩) (rawTerms := some (Proof.Events034.exact8761RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound8756.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8773.bound, LeftBound8756.bound]
def bound : CoeffClass := .finite ⟨1292337423279833362432, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8773.bound, LeftBound8756.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound8773.actual selector witness, LeftBound8756.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8945

namespace LeftBound8968
def owner : Owner := ⟨.program ⟨214⟩, ⟨99⟩⟩
def transferEvent : Nat := 8968
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 8967 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8967 .coefficient)
      LeftAuthority6440.bound (LeftAuthority6440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6440.derived selector witness)

def rawBound : CoeffClass := LeftAuthority6440.bound
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority6440.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound8968

namespace LeftBound8972
def owner : Owner := ⟨.program ⟨214⟩, ⟨12405⟩⟩
def transferEvent : Nat := 8972
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 8970 .coefficient) (.predecessor 1 8971 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8970 .coefficient)
      LeftAuthority165.bound (LeftAuthority165.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact166RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority165.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority165.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8971 .coefficient)
      LeftBound6447.bound (LeftBound6447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6447.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6447.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority165.bound LeftBound6447.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority165.bound, LeftBound6447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority165.actual selector witness) * (LeftBound6447.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound8972

namespace LeftBound8976
def owner : Owner := ⟨.program ⟨214⟩, ⟨6785⟩⟩
def transferEvent : Nat := 8976
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 8975 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8975 .coefficient)
      LeftAuthority5869.bound (LeftAuthority5869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5869.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5869.derived selector witness)

def rawBound : CoeffClass := LeftAuthority5869.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5869.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority5869.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound8976

namespace LeftBound8980
def owner : Owner := ⟨.program ⟨214⟩, ⟨7393⟩⟩
def transferEvent : Nat := 8980
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 8978 .coefficient) (.predecessor 1 8979 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8978 .coefficient)
      LeftBound6313.bound (LeftBound6313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8979 .coefficient)
      LeftBound8976.bound (LeftBound8976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8976.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound6313.bound LeftBound8976.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6313.bound, LeftBound8976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound6313.actual selector witness) * (LeftBound8976.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8980

namespace LeftBound8985
def owner : Owner := ⟨.program ⟨214⟩, ⟨12406⟩⟩
def transferEvent : Nat := 8985
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8983 .coefficient, .predecessor 1 8984 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8983 .coefficient)
      LeftBound8980.bound (LeftBound8980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8982RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8980.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8984 .coefficient)
      LeftBound8972.bound (LeftBound8972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8974RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8972.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8980.bound, LeftBound8972.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8980.bound, LeftBound8972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound8980.actual selector witness, LeftBound8972.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8985

namespace LeftBound8989
def owner : Owner := ⟨.program ⟨214⟩, ⟨12407⟩⟩
def transferEvent : Nat := 8989
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8987 .coefficient, .predecessor 1 8988 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8987 .coefficient)
      LeftBound8985.bound (LeftBound8985.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8986RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8985.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8985.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8988 .coefficient)
      LeftBound8968.bound (LeftBound8968.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8969RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8968.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8968.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8985.bound, LeftBound8968.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8985.bound, LeftBound8968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound8985.actual selector witness, LeftBound8968.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8989

namespace LeftBound8990
def owner : Owner := ⟨.program ⟨214⟩, ⟨12407⟩⟩
def transferEvent : Nat := 8990
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨99⟩⟩]⟩ [⟨.result 8969 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8969 .coefficient)
      LeftBound8968.bound (LeftBound8968.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨99⟩⟩) (rawTerms := some (Proof.Events035.exact8969RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8968.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8968.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound8968.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound8968.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound8990

namespace LeftBound8995
def owner : Owner := ⟨.program ⟨214⟩, ⟨12408⟩⟩
def transferEvent : Nat := 8995
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 8993 .coefficient) (.predecessor 1 8994 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8993 .coefficient)
      LeftBound8989.bound (LeftBound8989.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8992RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8989.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8989.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8994 .coefficient)
      LeftAuthority168.bound (LeftAuthority168.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact169RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority168.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority168.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound8989.bound LeftAuthority168.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8989.bound, LeftAuthority168.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound8989.actual selector witness) * (LeftAuthority168.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8995

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
