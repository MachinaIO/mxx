import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard699

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound101496
def owner : Owner := ⟨.program ⟨214⟩, ⟨10766⟩⟩
def transferEvent : Nat := 101496
def frameStart : Nat := 101449
def rule : BoundRule := .product (.predecessor 0 101494 .coefficient) (.predecessor 1 101495 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101494 .coefficient)
      LeftAuthority101492.bound (LeftAuthority101492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101493RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101492.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101492.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101495 .coefficient)
      LeftBound101490.bound (LeftBound101490.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101491RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101490.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101490.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority101492.bound LeftBound101490.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101492.bound, LeftBound101490.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority101492.actual selector witness) * (LeftBound101490.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101496

namespace LeftBound101512
def owner : Owner := ⟨.program ⟨214⟩, ⟨7835⟩⟩
def transferEvent : Nat := 101512
def frameStart : Nat := 101449
def rule : BoundRule := .scale (.predecessor 0 101510 .coefficient) (.value (.predecessor 1 101511 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101510 .coefficient)
      LeftAuthority101508.bound (LeftAuthority101508.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101509RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101508.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101508.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101511 .coefficient)
      LeftAuthority101499.bound (LeftAuthority101499.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority101499.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority101508.bound LeftAuthority101499.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101508.bound, LeftAuthority101499.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority101508.actual selector witness) * (LeftAuthority101499.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound101512

namespace LeftBound101515
def owner : Owner := ⟨.program ⟨214⟩, ⟨6782⟩⟩
def transferEvent : Nat := 101515
def frameStart : Nat := 101449
def rule : BoundRule := .identity (.predecessor 0 101514 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101514 .coefficient)
      LeftAuthority101502.bound (LeftAuthority101502.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101502.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101502.derived selector witness)

def rawBound : CoeffClass := LeftAuthority101502.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101502.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority101502.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound101515

namespace LeftBound101519
def owner : Owner := ⟨.program ⟨214⟩, ⟨7836⟩⟩
def transferEvent : Nat := 101519
def frameStart : Nat := 101449
def rule : BoundRule := .product (.predecessor 0 101517 .coefficient) (.predecessor 1 101518 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101517 .coefficient)
      LeftBound101515.bound (LeftBound101515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101515.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101518 .coefficient)
      LeftBound101512.bound (LeftBound101512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101512.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101515.bound LeftBound101512.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101515.bound, LeftBound101512.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101515.actual selector witness) * (LeftBound101512.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101519

namespace LeftBound101524
def owner : Owner := ⟨.program ⟨214⟩, ⟨10767⟩⟩
def transferEvent : Nat := 101524
def frameStart : Nat := 101449
def rule : BoundRule := .sum [.predecessor 0 101522 .coefficient, .predecessor 1 101523 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101522 .coefficient)
      LeftBound101519.bound (LeftBound101519.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101519.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101519.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101523 .coefficient)
      LeftBound101496.bound (LeftBound101496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101496.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101496.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound101519.bound, LeftBound101496.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101519.bound, LeftBound101496.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound101519.actual selector witness, LeftBound101496.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101524

namespace LeftBound101528
def owner : Owner := ⟨.program ⟨214⟩, ⟨24978⟩⟩
def transferEvent : Nat := 101528
def frameStart : Nat := 101449
def rule : BoundRule := .product (.predecessor 0 101526 .coefficient) (.predecessor 1 101527 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101526 .coefficient)
      LeftBound101524.bound (LeftBound101524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101524.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101524.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101527 .coefficient)
      LeftAuthority101481.bound (LeftAuthority101481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101481.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101481.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101524.bound LeftAuthority101481.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101524.bound, LeftAuthority101481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101524.actual selector witness) * (LeftAuthority101481.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101528

namespace LeftBound101539
def owner : Owner := ⟨.program ⟨214⟩, ⟨14945⟩⟩
def transferEvent : Nat := 101539
def frameStart : Nat := 101449
def rule : BoundRule := .product (.predecessor 0 101537 .coefficient) (.predecessor 1 101538 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101537 .coefficient)
      LeftAuthority101492.bound (LeftAuthority101492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101493RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101492.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101492.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101538 .coefficient)
      LeftAuthority101535.bound (LeftAuthority101535.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101536RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101535.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101535.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority101492.bound LeftAuthority101535.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101492.bound, LeftAuthority101535.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority101492.actual selector witness) * (LeftAuthority101535.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101539

namespace LeftBound101547
def owner : Owner := ⟨.program ⟨214⟩, ⟨14946⟩⟩
def transferEvent : Nat := 101547
def frameStart : Nat := 101449
def rule : BoundRule := .sum [.predecessor 0 101545 .coefficient, .predecessor 1 101546 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101545 .coefficient)
      LeftAuthority101543.bound (LeftAuthority101543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101544RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101543.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101543.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101546 .coefficient)
      LeftBound101539.bound (LeftBound101539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101541RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101539.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority101543.bound, LeftBound101539.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101543.bound, LeftBound101539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority101543.actual selector witness, LeftBound101539.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101547

namespace LeftBound101551
def owner : Owner := ⟨.program ⟨214⟩, ⟨24979⟩⟩
def transferEvent : Nat := 101551
def frameStart : Nat := 101449
def rule : BoundRule := .sum [.predecessor 0 101549 .coefficient, .predecessor 1 101550 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101549 .coefficient)
      LeftBound101547.bound (LeftBound101547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101547.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101547.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101550 .coefficient)
      LeftBound101528.bound (LeftBound101528.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101533RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101528.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101528.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound101547.bound, LeftBound101528.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101547.bound, LeftBound101528.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound101547.actual selector witness, LeftBound101528.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101551

namespace LeftBound101564
def owner : Owner := ⟨.program ⟨214⟩, ⟨24977⟩⟩
def transferEvent : Nat := 101564
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 101562 .coefficient, .predecessor 1 101563 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101562 .coefficient)
      LeftBound101409.bound (LeftBound101409.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101409.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101409.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101563 .coefficient)
      LeftBound101392.bound (LeftBound101392.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101399RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101392.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101392.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound101409.bound, LeftBound101392.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101409.bound, LeftBound101392.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound101409.actual selector witness, LeftBound101392.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101564

namespace LeftBound101567
def owner : Owner := ⟨.program ⟨214⟩, ⟨24977⟩⟩
def transferEvent : Nat := 101567
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 101561 .summary, .result 101399 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101561 .summary)
      LeftBound101411.bound (LeftBound101411.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19088⟩⟩) (rawTerms := some (Proof.Events396.exact101561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101411.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101399 .summary)
      LeftBound101394.bound (LeftBound101394.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24976⟩⟩) (rawTerms := some (Proof.Events396.exact101399RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101394.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound101411.bound, LeftBound101394.bound]
def bound : CoeffClass := .finite ⟨352014917316608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101411.bound, LeftBound101394.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound101411.actual selector witness, LeftBound101394.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101567

namespace LeftBound101571
def owner : Owner := ⟨.program ⟨214⟩, ⟨26531⟩⟩
def transferEvent : Nat := 101571
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 101569 .coefficient) (.predecessor 1 101570 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101569 .coefficient)
      LeftBound101564.bound (LeftBound101564.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101564.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101564.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101570 .coefficient)
      LeftAuthority101314.bound (LeftAuthority101314.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101315RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101314.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101314.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101564.bound LeftAuthority101314.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101564.bound, LeftAuthority101314.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101564.actual selector witness) * (LeftAuthority101314.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101571

namespace LeftBound101572
def owner : Owner := ⟨.program ⟨214⟩, ⟨26531⟩⟩
def transferEvent : Nat := 101572
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26529⟩⟩]⟩ [⟨.result 101315 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101315 .coefficient)
      LeftAuthority101314.bound (LeftAuthority101314.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26529⟩⟩) (rawTerms := some (Proof.Events395.exact101315RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101314.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101314.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority101314.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101314.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority101314.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound101572

namespace LeftBound101573
def owner : Owner := ⟨.program ⟨214⟩, ⟨26531⟩⟩
def transferEvent : Nat := 101573
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 101568 .summary) (.transfer 101572) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101568 .summary)
      LeftBound101567.bound (LeftBound101567.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24977⟩⟩) (rawTerms := some (Proof.Events396.exact101568RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101567.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 101572)
      LeftBound101572.bound (LeftBound101572.actual selector witness) := by
  exact .transfer (LeftBound101572.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101567.bound LeftBound101572.bound
def bound : CoeffClass := .finite ⟨1291900378790628425728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101567.bound, LeftBound101572.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101567.actual selector witness) * (LeftBound101572.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101573

namespace LeftBound101584
def owner : Owner := ⟨.program ⟨214⟩, ⟨20527⟩⟩
def transferEvent : Nat := 101584
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 101582 .coefficient) (.value (.predecessor 1 101583 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101582 .coefficient)
      LeftAuthority101580.bound (LeftAuthority101580.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101580.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101580.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101583 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority101580.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101580.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority101580.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound101584

namespace LeftBound101588
def owner : Owner := ⟨.program ⟨214⟩, ⟨20528⟩⟩
def transferEvent : Nat := 101588
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 101586 .coefficient) (.predecessor 1 101587 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101586 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101587 .coefficient)
      LeftBound101584.bound (LeftBound101584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101585RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101584.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound101584.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound101584.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound101584.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101588

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
