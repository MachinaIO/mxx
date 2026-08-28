import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard028
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard055

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound9908
def owner : Owner := ⟨.program ⟨214⟩, ⟨28787⟩⟩
def transferEvent : Nat := 9908
def frameStart : Nat := 9831
def rule : BoundRule := .product (.predecessor 0 9906 .coefficient) (.predecessor 1 9907 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9906 .coefficient)
      LeftBound9904.bound (LeftBound9904.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9905RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9904.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9904.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9907 .coefficient)
      LeftAuthority9881.bound (LeftAuthority9881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9882RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9881.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9881.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound9904.bound LeftAuthority9881.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9904.bound, LeftAuthority9881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound9904.actual selector witness) * (LeftAuthority9881.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9908

namespace LeftBound9919
def owner : Owner := ⟨.program ⟨214⟩, ⟨17133⟩⟩
def transferEvent : Nat := 9919
def frameStart : Nat := 9831
def rule : BoundRule := .product (.predecessor 0 9917 .coefficient) (.predecessor 1 9918 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9917 .coefficient)
      LeftAuthority9892.bound (LeftAuthority9892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9892.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9892.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9918 .coefficient)
      LeftAuthority9915.bound (LeftAuthority9915.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9916RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9915.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9915.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority9892.bound LeftAuthority9915.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9892.bound, LeftAuthority9915.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority9892.actual selector witness) * (LeftAuthority9915.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9919

namespace LeftBound9927
def owner : Owner := ⟨.program ⟨214⟩, ⟨17134⟩⟩
def transferEvent : Nat := 9927
def frameStart : Nat := 9831
def rule : BoundRule := .sum [.predecessor 0 9925 .coefficient, .predecessor 1 9926 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9925 .coefficient)
      LeftAuthority9923.bound (LeftAuthority9923.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9923.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9923.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9926 .coefficient)
      LeftBound9919.bound (LeftBound9919.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9919.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9919.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority9923.bound, LeftBound9919.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9923.bound, LeftBound9919.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority9923.actual selector witness, LeftBound9919.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9927

namespace LeftBound9931
def owner : Owner := ⟨.program ⟨214⟩, ⟨28791⟩⟩
def transferEvent : Nat := 9931
def frameStart : Nat := 9831
def rule : BoundRule := .sum [.predecessor 0 9929 .coefficient, .predecessor 1 9930 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9929 .coefficient)
      LeftBound9927.bound (LeftBound9927.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9927.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9927.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9930 .coefficient)
      LeftBound9908.bound (LeftBound9908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9913RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9908.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9908.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9927.bound, LeftBound9908.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9927.bound, LeftBound9908.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9927.actual selector witness, LeftBound9908.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9931

namespace LeftBound9944
def owner : Owner := ⟨.program ⟨214⟩, ⟨28789⟩⟩
def transferEvent : Nat := 9944
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9942 .coefficient, .predecessor 1 9943 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9942 .coefficient)
      LeftBound9773.bound (LeftBound9773.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9941RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9773.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9773.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9943 .coefficient)
      LeftBound9756.bound (LeftBound9756.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9763RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9756.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9756.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9773.bound, LeftBound9756.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9773.bound, LeftBound9756.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9773.actual selector witness, LeftBound9756.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9944

namespace LeftBound9947
def owner : Owner := ⟨.program ⟨214⟩, ⟨28789⟩⟩
def transferEvent : Nat := 9947
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 9941 .summary, .result 9763 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9941 .summary)
      LeftBound9775.bound (LeftBound9775.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21995⟩⟩) (rawTerms := some (Proof.Events038.exact9941RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound9775.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9763 .summary)
      LeftBound9758.bound (LeftBound9758.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28788⟩⟩) (rawTerms := some (Proof.Events038.exact9763RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound9758.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9775.bound, LeftBound9758.bound]
def bound : CoeffClass := .finite ⟨1292270185944771604480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9775.bound, LeftBound9758.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9775.actual selector witness, LeftBound9758.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9947

namespace LeftBound9970
def owner : Owner := ⟨.program ⟨214⟩, ⟨97⟩⟩
def transferEvent : Nat := 9970
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 9969 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9969 .coefficient)
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
end LeftBound9970

namespace LeftBound9974
def owner : Owner := ⟨.program ⟨214⟩, ⟨11796⟩⟩
def transferEvent : Nat := 9974
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 9972 .coefficient) (.predecessor 1 9973 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9972 .coefficient)
      LeftAuthority211.bound (LeftAuthority211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact212RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority211.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9973 .coefficient)
      LeftBound6447.bound (LeftBound6447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6447.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6447.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority211.bound LeftBound6447.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority211.bound, LeftBound6447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority211.actual selector witness) * (LeftBound6447.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound9974

namespace LeftBound9978
def owner : Owner := ⟨.program ⟨214⟩, ⟨6783⟩⟩
def transferEvent : Nat := 9978
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 9977 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9977 .coefficient)
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
end LeftBound9978

namespace LeftBound9982
def owner : Owner := ⟨.program ⟨214⟩, ⟨7391⟩⟩
def transferEvent : Nat := 9982
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9980 .coefficient) (.predecessor 1 9981 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9980 .coefficient)
      LeftBound6313.bound (LeftBound6313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9981 .coefficient)
      LeftBound9978.bound (LeftBound9978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9978.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound6313.bound LeftBound9978.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6313.bound, LeftBound9978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound6313.actual selector witness) * (LeftBound9978.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9982

namespace LeftBound9987
def owner : Owner := ⟨.program ⟨214⟩, ⟨11797⟩⟩
def transferEvent : Nat := 9987
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9985 .coefficient, .predecessor 1 9986 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9985 .coefficient)
      LeftBound9982.bound (LeftBound9982.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact9984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9982.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9982.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9986 .coefficient)
      LeftBound9974.bound (LeftBound9974.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9974.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9982.bound, LeftBound9974.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9982.bound, LeftBound9974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9982.actual selector witness, LeftBound9974.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9987

namespace LeftBound9991
def owner : Owner := ⟨.program ⟨214⟩, ⟨11798⟩⟩
def transferEvent : Nat := 9991
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9989 .coefficient, .predecessor 1 9990 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9989 .coefficient)
      LeftBound9987.bound (LeftBound9987.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact9988RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9987.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9987.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9990 .coefficient)
      LeftBound9970.bound (LeftBound9970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9971RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9970.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9987.bound, LeftBound9970.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9987.bound, LeftBound9970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9987.actual selector witness, LeftBound9970.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9991

namespace LeftBound9992
def owner : Owner := ⟨.program ⟨214⟩, ⟨11798⟩⟩
def transferEvent : Nat := 9992
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨97⟩⟩]⟩ [⟨.result 9971 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9971 .coefficient)
      LeftBound9970.bound (LeftBound9970.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨97⟩⟩) (rawTerms := some (Proof.Events038.exact9971RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9970.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound9970.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound9970.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound9992

namespace LeftBound9997
def owner : Owner := ⟨.program ⟨214⟩, ⟨11799⟩⟩
def transferEvent : Nat := 9997
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9995 .coefficient) (.predecessor 1 9996 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9995 .coefficient)
      LeftBound9991.bound (LeftBound9991.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact9994RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9991.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9991.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9996 .coefficient)
      LeftAuthority214.bound (LeftAuthority214.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority214.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority214.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound9991.bound LeftAuthority214.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9991.bound, LeftAuthority214.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound9991.actual selector witness) * (LeftAuthority214.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9997

namespace LeftBound9998
def owner : Owner := ⟨.program ⟨214⟩, ⟨11799⟩⟩
def transferEvent : Nat := 9998
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9630⟩⟩], []⟩ [⟨.result 215 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 215 .coefficient)
      LeftAuthority214.bound (LeftAuthority214.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9630⟩⟩) (rawTerms := some (Proof.Events000.exact215RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority214.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority214.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority214.bound []
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority214.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority214.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound9998

namespace LeftBound9999
def owner : Owner := ⟨.program ⟨214⟩, ⟨11799⟩⟩
def transferEvent : Nat := 9999
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 9994 .summary) (.transfer 9998) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9994 .summary)
      LeftBound9992.bound (LeftBound9992.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11798⟩⟩) (rawTerms := some (Proof.Events039.exact9994RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound9992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 9998)
      LeftBound9998.bound (LeftBound9998.actual selector witness) := by
  exact .transfer (LeftBound9998.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound9992.bound LeftBound9998.bound
def bound : CoeffClass := .finite ⟨24960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9992.bound, LeftBound9998.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound9992.actual selector witness) * (LeftBound9998.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9999

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
