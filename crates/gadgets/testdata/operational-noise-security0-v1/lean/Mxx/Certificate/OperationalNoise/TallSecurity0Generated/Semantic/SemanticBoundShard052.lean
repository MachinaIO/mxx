import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard028
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard051

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound9403
def owner : Owner := ⟨.program ⟨214⟩, ⟨16524⟩⟩
def transferEvent : Nat := 9403
def frameStart : Nat := 9330
def rule : BoundRule := .sum [.predecessor 0 9401 .coefficient, .predecessor 1 9402 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9401 .coefficient)
      LeftAuthority9399.bound (LeftAuthority9399.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9400RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9399.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9399.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9402 .coefficient)
      LeftBound9395.bound (LeftBound9395.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9397RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9395.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9395.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority9399.bound, LeftBound9395.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9399.bound, LeftBound9395.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority9399.actual selector witness, LeftBound9395.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9403

namespace LeftBound9407
def owner : Owner := ⟨.program ⟨214⟩, ⟨29004⟩⟩
def transferEvent : Nat := 9407
def frameStart : Nat := 9330
def rule : BoundRule := .product (.predecessor 0 9405 .coefficient) (.predecessor 1 9406 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9405 .coefficient)
      LeftBound9403.bound (LeftBound9403.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9404RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9403.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9403.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9406 .coefficient)
      LeftAuthority9380.bound (LeftAuthority9380.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9381RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9380.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9380.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound9403.bound LeftAuthority9380.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9403.bound, LeftAuthority9380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound9403.actual selector witness) * (LeftAuthority9380.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9407

namespace LeftBound9418
def owner : Owner := ⟨.program ⟨214⟩, ⟨17917⟩⟩
def transferEvent : Nat := 9418
def frameStart : Nat := 9330
def rule : BoundRule := .product (.predecessor 0 9416 .coefficient) (.predecessor 1 9417 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9416 .coefficient)
      LeftAuthority9391.bound (LeftAuthority9391.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9392RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9391.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9391.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9417 .coefficient)
      LeftAuthority9414.bound (LeftAuthority9414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9414.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9414.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority9391.bound LeftAuthority9414.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9391.bound, LeftAuthority9414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority9391.actual selector witness) * (LeftAuthority9414.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9418

namespace LeftBound9426
def owner : Owner := ⟨.program ⟨214⟩, ⟨17918⟩⟩
def transferEvent : Nat := 9426
def frameStart : Nat := 9330
def rule : BoundRule := .sum [.predecessor 0 9424 .coefficient, .predecessor 1 9425 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9424 .coefficient)
      LeftAuthority9422.bound (LeftAuthority9422.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9423RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9422.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9422.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9425 .coefficient)
      LeftBound9418.bound (LeftBound9418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9418.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority9422.bound, LeftBound9418.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9422.bound, LeftBound9418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority9422.actual selector witness, LeftBound9418.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9426

namespace LeftBound9430
def owner : Owner := ⟨.program ⟨214⟩, ⟨29008⟩⟩
def transferEvent : Nat := 9430
def frameStart : Nat := 9330
def rule : BoundRule := .sum [.predecessor 0 9428 .coefficient, .predecessor 1 9429 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9428 .coefficient)
      LeftBound9426.bound (LeftBound9426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9427RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9426.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9426.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9429 .coefficient)
      LeftBound9407.bound (LeftBound9407.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9412RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9407.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9407.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9426.bound, LeftBound9407.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9426.bound, LeftBound9407.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9426.actual selector witness, LeftBound9407.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9430

namespace LeftBound9443
def owner : Owner := ⟨.program ⟨214⟩, ⟨29006⟩⟩
def transferEvent : Nat := 9443
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9441 .coefficient, .predecessor 1 9442 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9441 .coefficient)
      LeftBound9272.bound (LeftBound9272.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9440RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9272.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9272.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9442 .coefficient)
      LeftBound9255.bound (LeftBound9255.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9262RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9255.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9255.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9272.bound, LeftBound9255.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9272.bound, LeftBound9255.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9272.actual selector witness, LeftBound9255.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9443

namespace LeftBound9446
def owner : Owner := ⟨.program ⟨214⟩, ⟨29006⟩⟩
def transferEvent : Nat := 9446
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 9440 .summary, .result 9262 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9440 .summary)
      LeftBound9274.bound (LeftBound9274.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22139⟩⟩) (rawTerms := some (Proof.Events036.exact9440RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound9274.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9262 .summary)
      LeftBound9257.bound (LeftBound9257.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29005⟩⟩) (rawTerms := some (Proof.Events036.exact9262RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound9257.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9274.bound, LeftBound9257.bound]
def bound : CoeffClass := .finite ⟨1292315010834812776448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9274.bound, LeftBound9257.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9274.actual selector witness, LeftBound9257.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9446

namespace LeftBound9469
def owner : Owner := ⟨.program ⟨214⟩, ⟨98⟩⟩
def transferEvent : Nat := 9469
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 9468 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9468 .coefficient)
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
end LeftBound9469

namespace LeftBound9473
def owner : Owner := ⟨.program ⟨214⟩, ⟨11992⟩⟩
def transferEvent : Nat := 9473
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 9471 .coefficient) (.predecessor 1 9472 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9471 .coefficient)
      LeftAuthority188.bound (LeftAuthority188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact189RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9472 .coefficient)
      LeftBound6447.bound (LeftBound6447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6447.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6447.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority188.bound LeftBound6447.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority188.bound, LeftBound6447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority188.actual selector witness) * (LeftBound6447.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound9473

namespace LeftBound9477
def owner : Owner := ⟨.program ⟨214⟩, ⟨6784⟩⟩
def transferEvent : Nat := 9477
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 9476 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9476 .coefficient)
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
end LeftBound9477

namespace LeftBound9481
def owner : Owner := ⟨.program ⟨214⟩, ⟨7392⟩⟩
def transferEvent : Nat := 9481
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9479 .coefficient) (.predecessor 1 9480 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9479 .coefficient)
      LeftBound6313.bound (LeftBound6313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9480 .coefficient)
      LeftBound9477.bound (LeftBound9477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9477.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound6313.bound LeftBound9477.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6313.bound, LeftBound9477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound6313.actual selector witness) * (LeftBound9477.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9481

namespace LeftBound9486
def owner : Owner := ⟨.program ⟨214⟩, ⟨11993⟩⟩
def transferEvent : Nat := 9486
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9484 .coefficient, .predecessor 1 9485 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9484 .coefficient)
      LeftBound9481.bound (LeftBound9481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9481.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9481.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9485 .coefficient)
      LeftBound9473.bound (LeftBound9473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9473.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9481.bound, LeftBound9473.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9481.bound, LeftBound9473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9481.actual selector witness, LeftBound9473.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9486

namespace LeftBound9490
def owner : Owner := ⟨.program ⟨214⟩, ⟨11994⟩⟩
def transferEvent : Nat := 9490
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9488 .coefficient, .predecessor 1 9489 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9488 .coefficient)
      LeftBound9486.bound (LeftBound9486.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9487RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9486.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9486.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9489 .coefficient)
      LeftBound9469.bound (LeftBound9469.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9470RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9469.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9469.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9486.bound, LeftBound9469.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9486.bound, LeftBound9469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9486.actual selector witness, LeftBound9469.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9490

namespace LeftBound9491
def owner : Owner := ⟨.program ⟨214⟩, ⟨11994⟩⟩
def transferEvent : Nat := 9491
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨98⟩⟩]⟩ [⟨.result 9470 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9470 .coefficient)
      LeftBound9469.bound (LeftBound9469.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨98⟩⟩) (rawTerms := some (Proof.Events036.exact9470RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9469.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9469.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound9469.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound9469.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound9491

namespace LeftBound9496
def owner : Owner := ⟨.program ⟨214⟩, ⟨11995⟩⟩
def transferEvent : Nat := 9496
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9494 .coefficient) (.predecessor 1 9495 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9494 .coefficient)
      LeftBound9490.bound (LeftBound9490.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9493RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9490.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9490.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9495 .coefficient)
      LeftAuthority191.bound (LeftAuthority191.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact192RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority191.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority191.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound9490.bound LeftAuthority191.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9490.bound, LeftAuthority191.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound9490.actual selector witness) * (LeftAuthority191.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9496

namespace LeftBound9497
def owner : Owner := ⟨.program ⟨214⟩, ⟨11995⟩⟩
def transferEvent : Nat := 9497
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9735⟩⟩], []⟩ [⟨.result 192 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 192 .coefficient)
      LeftAuthority191.bound (LeftAuthority191.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9735⟩⟩) (rawTerms := some (Proof.Events000.exact192RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority191.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority191.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority191.bound []
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority191.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority191.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound9497

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
