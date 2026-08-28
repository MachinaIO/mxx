import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard064
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard370

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound54955
def owner : Owner := ⟨.program ⟨214⟩, ⟨16225⟩⟩
def transferEvent : Nat := 54955
def frameStart : Nat := 54882
def rule : BoundRule := .sum [.predecessor 0 54953 .coefficient, .predecessor 1 54954 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54953 .coefficient)
      LeftAuthority54951.bound (LeftAuthority54951.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54952RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54951.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54951.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54954 .coefficient)
      LeftBound54947.bound (LeftBound54947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54949RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54947.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54947.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority54951.bound, LeftBound54947.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54951.bound, LeftBound54947.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority54951.actual selector witness, LeftBound54947.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54955

namespace LeftBound54959
def owner : Owner := ⟨.program ⟨214⟩, ⟨28314⟩⟩
def transferEvent : Nat := 54959
def frameStart : Nat := 54882
def rule : BoundRule := .product (.predecessor 0 54957 .coefficient) (.predecessor 1 54958 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54957 .coefficient)
      LeftBound54955.bound (LeftBound54955.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54956RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54955.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54955.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54958 .coefficient)
      LeftAuthority54932.bound (LeftAuthority54932.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54933RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54932.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54932.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound54955.bound LeftAuthority54932.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54955.bound, LeftAuthority54932.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound54955.actual selector witness) * (LeftAuthority54932.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54959

namespace LeftBound54970
def owner : Owner := ⟨.program ⟨214⟩, ⟨18364⟩⟩
def transferEvent : Nat := 54970
def frameStart : Nat := 54882
def rule : BoundRule := .product (.predecessor 0 54968 .coefficient) (.predecessor 1 54969 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54968 .coefficient)
      LeftAuthority54943.bound (LeftAuthority54943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54943.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54969 .coefficient)
      LeftAuthority54966.bound (LeftAuthority54966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54966.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54966.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority54943.bound LeftAuthority54966.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54943.bound, LeftAuthority54966.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority54943.actual selector witness) * (LeftAuthority54966.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54970

namespace LeftBound54978
def owner : Owner := ⟨.program ⟨214⟩, ⟨18365⟩⟩
def transferEvent : Nat := 54978
def frameStart : Nat := 54882
def rule : BoundRule := .sum [.predecessor 0 54976 .coefficient, .predecessor 1 54977 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54976 .coefficient)
      LeftAuthority54974.bound (LeftAuthority54974.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54974.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54974.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54977 .coefficient)
      LeftBound54970.bound (LeftBound54970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54972RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54970.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority54974.bound, LeftBound54970.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54974.bound, LeftBound54970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority54974.actual selector witness, LeftBound54970.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54978

namespace LeftBound54982
def owner : Owner := ⟨.program ⟨214⟩, ⟨28318⟩⟩
def transferEvent : Nat := 54982
def frameStart : Nat := 54882
def rule : BoundRule := .sum [.predecessor 0 54980 .coefficient, .predecessor 1 54981 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54980 .coefficient)
      LeftBound54978.bound (LeftBound54978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54978.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54981 .coefficient)
      LeftBound54959.bound (LeftBound54959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54959.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54959.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54978.bound, LeftBound54959.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54978.bound, LeftBound54959.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54978.actual selector witness, LeftBound54959.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54982

namespace LeftBound54995
def owner : Owner := ⟨.program ⟨214⟩, ⟨28316⟩⟩
def transferEvent : Nat := 54995
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54993 .coefficient, .predecessor 1 54994 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54993 .coefficient)
      LeftBound54824.bound (LeftBound54824.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54992RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54824.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54824.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54994 .coefficient)
      LeftBound54807.bound (LeftBound54807.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54814RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54807.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54807.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54824.bound, LeftBound54807.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54824.bound, LeftBound54807.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54824.actual selector witness, LeftBound54807.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54995

namespace LeftBound54998
def owner : Owner := ⟨.program ⟨214⟩, ⟨28316⟩⟩
def transferEvent : Nat := 54998
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 54992 .summary, .result 54814 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54992 .summary)
      LeftBound54826.bound (LeftBound54826.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21695⟩⟩) (rawTerms := some (Proof.Events214.exact54992RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54826.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54814 .summary)
      LeftBound54809.bound (LeftBound54809.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28315⟩⟩) (rawTerms := some (Proof.Events214.exact54814RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54809.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54826.bound, LeftBound54809.bound]
def bound : CoeffClass := .finite ⟨1292180536164689260544, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54826.bound, LeftBound54809.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54826.actual selector witness, LeftBound54809.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54998

namespace LeftBound55022
def owner : Owner := ⟨.program ⟨214⟩, ⟨11558⟩⟩
def transferEvent : Nat := 55022
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 55020 .coefficient) (.predecessor 1 55021 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55020 .coefficient)
      LeftAuthority2544.bound (LeftAuthority2544.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2545RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2544.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2544.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55021 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2544.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2544.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2544.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound55022

namespace LeftBound55027
def owner : Owner := ⟨.program ⟨214⟩, ⟨7274⟩⟩
def transferEvent : Nat := 55027
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 55025 .coefficient) (.predecessor 1 55026 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55025 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55026 .coefficient)
      LeftBound10980.bound (LeftBound10980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10980.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound10980.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound10980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound10980.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55027

namespace LeftBound55032
def owner : Owner := ⟨.program ⟨214⟩, ⟨11559⟩⟩
def transferEvent : Nat := 55032
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55030 .coefficient, .predecessor 1 55031 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55030 .coefficient)
      LeftBound55027.bound (LeftBound55027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact55029RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55027.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55031 .coefficient)
      LeftBound55022.bound (LeftBound55022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact55024RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55022.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55022.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55027.bound, LeftBound55022.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55027.bound, LeftBound55022.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55027.actual selector witness, LeftBound55022.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55032

namespace LeftBound55036
def owner : Owner := ⟨.program ⟨214⟩, ⟨11560⟩⟩
def transferEvent : Nat := 55036
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55034 .coefficient, .predecessor 1 55035 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55034 .coefficient)
      LeftBound55032.bound (LeftBound55032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact55033RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55032.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55035 .coefficient)
      LeftBound10972.bound (LeftBound10972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10972.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55032.bound, LeftBound10972.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55032.bound, LeftBound10972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55032.actual selector witness, LeftBound10972.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55036

namespace LeftBound55037
def owner : Owner := ⟨.program ⟨214⟩, ⟨11560⟩⟩
def transferEvent : Nat := 55037
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨94⟩⟩]⟩ [⟨.result 10973 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10973 .coefficient)
      LeftBound10972.bound (LeftBound10972.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨94⟩⟩) (rawTerms := some (Proof.Events042.exact10973RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10972.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound10972.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound10972.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound55037

namespace LeftBound55042
def owner : Owner := ⟨.program ⟨214⟩, ⟨14436⟩⟩
def transferEvent : Nat := 55042
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 55040 .coefficient) (.predecessor 1 55041 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55040 .coefficient)
      LeftBound55036.bound (LeftBound55036.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact55039RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55036.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55036.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55041 .coefficient)
      LeftAuthority2547.bound (LeftAuthority2547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2547.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2547.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound55036.bound LeftAuthority2547.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55036.bound, LeftAuthority2547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound55036.actual selector witness) * (LeftAuthority2547.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55042

namespace LeftBound55043
def owner : Owner := ⟨.program ⟨214⟩, ⟨14436⟩⟩
def transferEvent : Nat := 55043
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨14433⟩⟩], []⟩ [⟨.result 2548 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2548 .coefficient)
      LeftAuthority2547.bound (LeftAuthority2547.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨14433⟩⟩) (rawTerms := some (Proof.Events009.exact2548RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2547.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2547.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority2547.bound []
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority2547.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound55043

namespace LeftBound55044
def owner : Owner := ⟨.program ⟨214⟩, ⟨14436⟩⟩
def transferEvent : Nat := 55044
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 55039 .summary) (.transfer 55043) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55039 .summary)
      LeftBound55037.bound (LeftBound55037.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11560⟩⟩) (rawTerms := some (Proof.Events214.exact55039RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55037.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 55043)
      LeftBound55043.bound (LeftBound55043.actual selector witness) := by
  exact .transfer (LeftBound55043.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound55037.bound LeftBound55043.bound
def bound : CoeffClass := .finite ⟨18304, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55037.bound, LeftBound55043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound55037.actual selector witness) * (LeftBound55043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55044

namespace LeftBound55050
def owner : Owner := ⟨.program ⟨214⟩, ⟨14437⟩⟩
def transferEvent : Nat := 55050
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 55048 .coefficient) (.predecessor 1 55049 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55048 .coefficient)
      LeftAuthority2547.bound (LeftAuthority2547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2547.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2547.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55049 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2547.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2547.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2547.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound55050

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
