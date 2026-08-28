import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard091
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard598

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound87513
def owner : Owner := ⟨.program ⟨214⟩, ⟨15115⟩⟩
def transferEvent : Nat := 87513
def frameStart : Nat := 87474
def rule : BoundRule := .identity (.predecessor 0 87512 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87512 .coefficient)
      LeftAuthority87510.bound (LeftAuthority87510.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87511RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87510.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87510.derived selector witness)

def rawBound : CoeffClass := LeftAuthority87510.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87510.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority87510.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound87513

namespace LeftBound87530
def owner : Owner := ⟨.program ⟨214⟩, ⟨15154⟩⟩
def transferEvent : Nat := 87530
def frameStart : Nat := 87474
def rule : BoundRule := .sum [.predecessor 0 87528 .coefficient, .predecessor 1 87529 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87528 .coefficient)
      LeftBound87513.bound (LeftBound87513.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound87513.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87529 .coefficient)
      LeftAuthority87526.bound (LeftAuthority87526.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority87526.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87513.bound, LeftAuthority87526.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87513.bound, LeftAuthority87526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87513.actual selector witness, LeftAuthority87526.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87530

namespace LeftBound87533
def owner : Owner := ⟨.program ⟨214⟩, ⟨15155⟩⟩
def transferEvent : Nat := 87533
def frameStart : Nat := 87474
def rule : BoundRule := .identity (.predecessor 0 87532 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87532 .coefficient)
      LeftBound87530.bound (LeftBound87530.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound87530.derived selector witness)

def rawBound : CoeffClass := LeftBound87530.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87530.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound87530.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound87533

namespace LeftBound87539
def owner : Owner := ⟨.program ⟨214⟩, ⟨15156⟩⟩
def transferEvent : Nat := 87539
def frameStart : Nat := 87474
def rule : BoundRule := .product (.predecessor 0 87537 .coefficient) (.predecessor 1 87538 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87537 .coefficient)
      LeftAuthority87535.bound (LeftAuthority87535.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87536RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87535.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87535.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87538 .coefficient)
      LeftBound87533.bound (LeftBound87533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87534RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87533.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87533.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority87535.bound LeftBound87533.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87535.bound, LeftBound87533.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority87535.actual selector witness) * (LeftBound87533.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87539

namespace LeftBound87547
def owner : Owner := ⟨.program ⟨214⟩, ⟨15157⟩⟩
def transferEvent : Nat := 87547
def frameStart : Nat := 87474
def rule : BoundRule := .sum [.predecessor 0 87545 .coefficient, .predecessor 1 87546 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87545 .coefficient)
      LeftAuthority87543.bound (LeftAuthority87543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87544RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87543.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87543.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87546 .coefficient)
      LeftBound87539.bound (LeftBound87539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87541RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87539.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority87543.bound, LeftBound87539.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87543.bound, LeftBound87539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority87543.actual selector witness, LeftBound87539.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87547

namespace LeftBound87551
def owner : Owner := ⟨.program ⟨214⟩, ⟨26782⟩⟩
def transferEvent : Nat := 87551
def frameStart : Nat := 87474
def rule : BoundRule := .product (.predecessor 0 87549 .coefficient) (.predecessor 1 87550 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87549 .coefficient)
      LeftBound87547.bound (LeftBound87547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87547.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87547.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87550 .coefficient)
      LeftAuthority87524.bound (LeftAuthority87524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87524.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87524.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound87547.bound LeftAuthority87524.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87547.bound, LeftAuthority87524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound87547.actual selector witness) * (LeftAuthority87524.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87551

namespace LeftBound87562
def owner : Owner := ⟨.program ⟨214⟩, ⟨15368⟩⟩
def transferEvent : Nat := 87562
def frameStart : Nat := 87474
def rule : BoundRule := .product (.predecessor 0 87560 .coefficient) (.predecessor 1 87561 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87560 .coefficient)
      LeftAuthority87535.bound (LeftAuthority87535.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87536RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87535.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87535.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87561 .coefficient)
      LeftAuthority87558.bound (LeftAuthority87558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87559RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87558.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87558.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority87535.bound LeftAuthority87558.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87535.bound, LeftAuthority87558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority87535.actual selector witness) * (LeftAuthority87558.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87562

namespace LeftBound87570
def owner : Owner := ⟨.program ⟨214⟩, ⟨15369⟩⟩
def transferEvent : Nat := 87570
def frameStart : Nat := 87474
def rule : BoundRule := .sum [.predecessor 0 87568 .coefficient, .predecessor 1 87569 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87568 .coefficient)
      LeftAuthority87566.bound (LeftAuthority87566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87567RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87566.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87566.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87569 .coefficient)
      LeftBound87562.bound (LeftBound87562.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87564RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87562.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87562.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority87566.bound, LeftBound87562.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87566.bound, LeftBound87562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority87566.actual selector witness, LeftBound87562.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87570

namespace LeftBound87574
def owner : Owner := ⟨.program ⟨214⟩, ⟨26786⟩⟩
def transferEvent : Nat := 87574
def frameStart : Nat := 87474
def rule : BoundRule := .sum [.predecessor 0 87572 .coefficient, .predecessor 1 87573 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87572 .coefficient)
      LeftBound87570.bound (LeftBound87570.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87571RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87570.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87570.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87573 .coefficient)
      LeftBound87551.bound (LeftBound87551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87556RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87551.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87551.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87570.bound, LeftBound87551.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87570.bound, LeftBound87551.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87570.actual selector witness, LeftBound87551.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87574

namespace LeftBound87587
def owner : Owner := ⟨.program ⟨214⟩, ⟨26784⟩⟩
def transferEvent : Nat := 87587
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 87585 .coefficient, .predecessor 1 87586 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87585 .coefficient)
      LeftBound87416.bound (LeftBound87416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87416.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87416.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87586 .coefficient)
      LeftBound87399.bound (LeftBound87399.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87399.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87399.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87416.bound, LeftBound87399.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87416.bound, LeftBound87399.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87416.actual selector witness, LeftBound87399.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87587

namespace LeftBound87590
def owner : Owner := ⟨.program ⟨214⟩, ⟨26784⟩⟩
def transferEvent : Nat := 87590
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 87584 .summary, .result 87406 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87584 .summary)
      LeftBound87418.bound (LeftBound87418.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20683⟩⟩) (rawTerms := some (Proof.Events342.exact87584RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87418.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87406 .summary)
      LeftBound87401.bound (LeftBound87401.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26783⟩⟩) (rawTerms := some (Proof.Events341.exact87406RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87401.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87418.bound, LeftBound87401.bound]
def bound : CoeffClass := .finite ⟨1291911586824442228736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87418.bound, LeftBound87401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87418.actual selector witness, LeftBound87401.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87590

namespace LeftBound87614
def owner : Owner := ⟨.program ⟨214⟩, ⟨10679⟩⟩
def transferEvent : Nat := 87614
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 87612 .coefficient) (.predecessor 1 87613 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87612 .coefficient)
      LeftAuthority4195.bound (LeftAuthority4195.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4196RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4195.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4195.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87613 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4195.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4195.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4195.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound87614

namespace LeftBound87619
def owner : Owner := ⟨.program ⟨214⟩, ⟨7229⟩⟩
def transferEvent : Nat := 87619
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 87617 .coefficient) (.predecessor 1 87618 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87617 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87618 .coefficient)
      LeftBound14487.bound (LeftBound14487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14487.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14487.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound14487.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound14487.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound14487.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87619

namespace LeftBound87624
def owner : Owner := ⟨.program ⟨214⟩, ⟨10680⟩⟩
def transferEvent : Nat := 87624
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 87622 .coefficient, .predecessor 1 87623 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87622 .coefficient)
      LeftBound87619.bound (LeftBound87619.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87621RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87619.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87619.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87623 .coefficient)
      LeftBound87614.bound (LeftBound87614.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87614.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87614.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87619.bound, LeftBound87614.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87619.bound, LeftBound87614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87619.actual selector witness, LeftBound87614.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87624

namespace LeftBound87628
def owner : Owner := ⟨.program ⟨214⟩, ⟨10681⟩⟩
def transferEvent : Nat := 87628
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 87626 .coefficient, .predecessor 1 87627 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87626 .coefficient)
      LeftBound87624.bound (LeftBound87624.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87624.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87624.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87627 .coefficient)
      LeftBound14479.bound (LeftBound14479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14479.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87624.bound, LeftBound14479.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87624.bound, LeftBound14479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87624.actual selector witness, LeftBound14479.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87628

namespace LeftBound87629
def owner : Owner := ⟨.program ⟨214⟩, ⟨10681⟩⟩
def transferEvent : Nat := 87629
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨87⟩⟩]⟩ [⟨.result 14480 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14480 .coefficient)
      LeftBound14479.bound (LeftBound14479.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨87⟩⟩) (rawTerms := some (Proof.Events056.exact14480RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14479.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound14479.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound14479.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound87629

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
