import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard568

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound83503
def owner : Owner := ⟨.program ⟨214⟩, ⟨6763⟩⟩
def transferEvent : Nat := 83503
def frameStart : Nat := 83427
def rule : BoundRule := .identity (.predecessor 0 83502 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83502 .coefficient)
      LeftAuthority83490.bound (LeftAuthority83490.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83491RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83490.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83490.derived selector witness)

def rawBound : CoeffClass := LeftAuthority83490.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83490.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority83490.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound83503

namespace LeftBound83507
def owner : Owner := ⟨.program ⟨214⟩, ⟨7863⟩⟩
def transferEvent : Nat := 83507
def frameStart : Nat := 83427
def rule : BoundRule := .product (.predecessor 0 83505 .coefficient) (.predecessor 1 83506 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83505 .coefficient)
      LeftBound83503.bound (LeftBound83503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83503.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83506 .coefficient)
      LeftBound83500.bound (LeftBound83500.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83501RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83500.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83500.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound83503.bound LeftBound83500.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83503.bound, LeftBound83500.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound83503.actual selector witness) * (LeftBound83500.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83507

namespace LeftBound83512
def owner : Owner := ⟨.program ⟨214⟩, ⟨11860⟩⟩
def transferEvent : Nat := 83512
def frameStart : Nat := 83427
def rule : BoundRule := .sum [.predecessor 0 83510 .coefficient, .predecessor 1 83511 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83510 .coefficient)
      LeftBound83507.bound (LeftBound83507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83509RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83507.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83507.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83511 .coefficient)
      LeftBound83486.bound (LeftBound83486.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83486.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83486.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83507.bound, LeftBound83486.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83507.bound, LeftBound83486.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83507.actual selector witness, LeftBound83486.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83512

namespace LeftBound83516
def owner : Owner := ⟨.program ⟨214⟩, ⟨25145⟩⟩
def transferEvent : Nat := 83516
def frameStart : Nat := 83427
def rule : BoundRule := .product (.predecessor 0 83514 .coefficient) (.predecessor 1 83515 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83514 .coefficient)
      LeftBound83512.bound (LeftBound83512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83512.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83515 .coefficient)
      LeftAuthority83471.bound (LeftAuthority83471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83471.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83471.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound83512.bound LeftAuthority83471.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83512.bound, LeftAuthority83471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound83512.actual selector witness) * (LeftAuthority83471.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83516

namespace LeftBound83527
def owner : Owner := ⟨.program ⟨214⟩, ⟨16264⟩⟩
def transferEvent : Nat := 83527
def frameStart : Nat := 83427
def rule : BoundRule := .product (.predecessor 0 83525 .coefficient) (.predecessor 1 83526 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83525 .coefficient)
      LeftAuthority83482.bound (LeftAuthority83482.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83482.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83482.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83526 .coefficient)
      LeftAuthority83523.bound (LeftAuthority83523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83524RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83523.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83523.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority83482.bound LeftAuthority83523.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83482.bound, LeftAuthority83523.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority83482.actual selector witness) * (LeftAuthority83523.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83527

namespace LeftBound83535
def owner : Owner := ⟨.program ⟨214⟩, ⟨16265⟩⟩
def transferEvent : Nat := 83535
def frameStart : Nat := 83427
def rule : BoundRule := .sum [.predecessor 0 83533 .coefficient, .predecessor 1 83534 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83533 .coefficient)
      LeftAuthority83531.bound (LeftAuthority83531.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83532RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83531.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83531.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83534 .coefficient)
      LeftBound83527.bound (LeftBound83527.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83529RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83527.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83527.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority83531.bound, LeftBound83527.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83531.bound, LeftBound83527.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority83531.actual selector witness, LeftBound83527.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83535

namespace LeftBound83539
def owner : Owner := ⟨.program ⟨214⟩, ⟨25146⟩⟩
def transferEvent : Nat := 83539
def frameStart : Nat := 83427
def rule : BoundRule := .sum [.predecessor 0 83537 .coefficient, .predecessor 1 83538 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83537 .coefficient)
      LeftBound83535.bound (LeftBound83535.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83536RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83535.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83535.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83538 .coefficient)
      LeftBound83516.bound (LeftBound83516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83516.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83535.bound, LeftBound83516.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83535.bound, LeftBound83516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83535.actual selector witness, LeftBound83516.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83539

namespace LeftBound83552
def owner : Owner := ⟨.program ⟨214⟩, ⟨25144⟩⟩
def transferEvent : Nat := 83552
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 83550 .coefficient, .predecessor 1 83551 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83550 .coefficient)
      LeftBound83375.bound (LeftBound83375.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83549RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83375.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83375.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83551 .coefficient)
      LeftBound83358.bound (LeftBound83358.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83365RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83358.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83358.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83375.bound, LeftBound83358.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83375.bound, LeftBound83358.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83375.actual selector witness, LeftBound83358.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83552

namespace LeftBound83555
def owner : Owner := ⟨.program ⟨214⟩, ⟨25144⟩⟩
def transferEvent : Nat := 83555
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 83549 .summary, .result 83365 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83549 .summary)
      LeftBound83377.bound (LeftBound83377.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19747⟩⟩) (rawTerms := some (Proof.Events326.exact83549RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83377.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83365 .summary)
      LeftBound83360.bound (LeftBound83360.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25143⟩⟩) (rawTerms := some (Proof.Events325.exact83365RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83360.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83377.bound, LeftBound83360.bound]
def bound : CoeffClass := .finite ⟨352097360556032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83377.bound, LeftBound83360.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83377.actual selector witness, LeftBound83360.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83555

namespace LeftBound83559
def owner : Owner := ⟨.program ⟨214⟩, ⟨28519⟩⟩
def transferEvent : Nat := 83559
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 83557 .coefficient) (.predecessor 1 83558 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83557 .coefficient)
      LeftBound83552.bound (LeftBound83552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83556RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83552.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83552.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83558 .coefficient)
      LeftAuthority83280.bound (LeftAuthority83280.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events325.exact83281RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83280.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83280.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound83552.bound LeftAuthority83280.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83552.bound, LeftAuthority83280.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound83552.actual selector witness) * (LeftAuthority83280.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83559

namespace LeftBound83560
def owner : Owner := ⟨.program ⟨214⟩, ⟨28519⟩⟩
def transferEvent : Nat := 83560
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩ [⟨.result 83281 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83281 .coefficient)
      LeftAuthority83280.bound (LeftAuthority83280.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28517⟩⟩) (rawTerms := some (Proof.Events325.exact83281RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83280.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83280.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority83280.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83280.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority83280.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound83560

namespace LeftBound83561
def owner : Owner := ⟨.program ⟨214⟩, ⟨28519⟩⟩
def transferEvent : Nat := 83561
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 83556 .summary) (.transfer 83560) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83556 .summary)
      LeftBound83555.bound (LeftBound83555.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25144⟩⟩) (rawTerms := some (Proof.Events326.exact83556RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83555.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 83560)
      LeftBound83560.bound (LeftBound83560.actual selector witness) := by
  exact .transfer (LeftBound83560.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound83555.bound LeftBound83560.bound
def bound : CoeffClass := .finite ⟨1292202946798406336512, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83555.bound, LeftBound83560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound83555.actual selector witness) * (LeftBound83560.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83561

namespace LeftBound83572
def owner : Owner := ⟨.program ⟨214⟩, ⟨21834⟩⟩
def transferEvent : Nat := 83572
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 83570 .coefficient) (.value (.predecessor 1 83571 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83570 .coefficient)
      LeftAuthority83568.bound (LeftAuthority83568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83569RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83568.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83568.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83571 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority83568.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83568.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority83568.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound83572

namespace LeftBound83576
def owner : Owner := ⟨.program ⟨214⟩, ⟨21835⟩⟩
def transferEvent : Nat := 83576
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 83574 .coefficient) (.predecessor 1 83575 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83574 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83575 .coefficient)
      LeftBound83572.bound (LeftBound83572.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83572.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83572.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound83572.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound83572.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound83572.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83576

namespace LeftBound83577
def owner : Owner := ⟨.program ⟨214⟩, ⟨21835⟩⟩
def transferEvent : Nat := 83577
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21832⟩⟩]⟩ [⟨.result 83569 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83569 .coefficient)
      LeftAuthority83568.bound (LeftAuthority83568.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21832⟩⟩) (rawTerms := some (Proof.Events326.exact83569RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83568.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83568.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority83568.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83568.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority83568.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound83577

namespace LeftBound83578
def owner : Owner := ⟨.program ⟨214⟩, ⟨21835⟩⟩
def transferEvent : Nat := 83578
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 83577) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 83577)
      LeftBound83577.bound (LeftBound83577.actual selector witness) := by
  exact .transfer (LeftBound83577.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound83577.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound83577.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound83577.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83578

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
