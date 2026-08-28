import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard210

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound32381
def owner : Owner := ⟨.program ⟨214⟩, ⟨22351⟩⟩
def transferEvent : Nat := 32381
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22348⟩⟩]⟩ [⟨.result 32373 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 32373 .coefficient)
      LeftAuthority32372.bound (LeftAuthority32372.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22348⟩⟩) (rawTerms := some (Proof.Events126.exact32373RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32372.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32372.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority32372.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32372.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority32372.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound32381

namespace LeftBound32382
def owner : Owner := ⟨.program ⟨214⟩, ⟨22351⟩⟩
def transferEvent : Nat := 32382
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 32381) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 32381)
      LeftBound32381.bound (LeftBound32381.actual selector witness) := by
  exact .transfer (LeftBound32381.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound32381.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound32381.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound32381.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32382

namespace LeftBound32477
def owner : Owner := ⟨.program ⟨214⟩, ⟨16646⟩⟩
def transferEvent : Nat := 32477
def frameStart : Nat := 32438
def rule : BoundRule := .identity (.predecessor 0 32476 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32476 .coefficient)
      LeftAuthority32474.bound (LeftAuthority32474.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32474.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32474.derived selector witness)

def rawBound : CoeffClass := LeftAuthority32474.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32474.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority32474.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound32477

namespace LeftBound32494
def owner : Owner := ⟨.program ⟨214⟩, ⟨16720⟩⟩
def transferEvent : Nat := 32494
def frameStart : Nat := 32438
def rule : BoundRule := .sum [.predecessor 0 32492 .coefficient, .predecessor 1 32493 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32492 .coefficient)
      LeftBound32477.bound (LeftBound32477.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound32477.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32493 .coefficient)
      LeftAuthority32490.bound (LeftAuthority32490.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority32490.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound32477.bound, LeftAuthority32490.bound]
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32477.bound, LeftAuthority32490.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound32477.actual selector witness, LeftAuthority32490.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32494

namespace LeftBound32497
def owner : Owner := ⟨.program ⟨214⟩, ⟨16721⟩⟩
def transferEvent : Nat := 32497
def frameStart : Nat := 32438
def rule : BoundRule := .identity (.predecessor 0 32496 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32496 .coefficient)
      LeftBound32494.bound (LeftBound32494.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound32494.derived selector witness)

def rawBound : CoeffClass := LeftBound32494.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32494.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound32494.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound32497

namespace LeftBound32503
def owner : Owner := ⟨.program ⟨214⟩, ⟨16722⟩⟩
def transferEvent : Nat := 32503
def frameStart : Nat := 32438
def rule : BoundRule := .product (.predecessor 0 32501 .coefficient) (.predecessor 1 32502 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32501 .coefficient)
      LeftAuthority32499.bound (LeftAuthority32499.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32499.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32499.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32502 .coefficient)
      LeftBound32497.bound (LeftBound32497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32497.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32497.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority32499.bound LeftBound32497.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32499.bound, LeftBound32497.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority32499.actual selector witness) * (LeftBound32497.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32503

namespace LeftBound32511
def owner : Owner := ⟨.program ⟨214⟩, ⟨16723⟩⟩
def transferEvent : Nat := 32511
def frameStart : Nat := 32438
def rule : BoundRule := .sum [.predecessor 0 32509 .coefficient, .predecessor 1 32510 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32509 .coefficient)
      LeftAuthority32507.bound (LeftAuthority32507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32507.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32507.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32510 .coefficient)
      LeftBound32503.bound (LeftBound32503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32505RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32503.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority32507.bound, LeftBound32503.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32507.bound, LeftBound32503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority32507.actual selector witness, LeftBound32503.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32511

namespace LeftBound32515
def owner : Owner := ⟨.program ⟨214⟩, ⟨29418⟩⟩
def transferEvent : Nat := 32515
def frameStart : Nat := 32438
def rule : BoundRule := .product (.predecessor 0 32513 .coefficient) (.predecessor 1 32514 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32513 .coefficient)
      LeftBound32511.bound (LeftBound32511.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32511.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32511.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32514 .coefficient)
      LeftAuthority32488.bound (LeftAuthority32488.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32489RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32488.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32488.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound32511.bound LeftAuthority32488.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32511.bound, LeftAuthority32488.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound32511.actual selector witness) * (LeftAuthority32488.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32515

namespace LeftBound32526
def owner : Owner := ⟨.program ⟨214⟩, ⟨17732⟩⟩
def transferEvent : Nat := 32526
def frameStart : Nat := 32438
def rule : BoundRule := .product (.predecessor 0 32524 .coefficient) (.predecessor 1 32525 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32524 .coefficient)
      LeftAuthority32499.bound (LeftAuthority32499.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32499.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32499.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32525 .coefficient)
      LeftAuthority32522.bound (LeftAuthority32522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32522.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32522.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority32499.bound LeftAuthority32522.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32499.bound, LeftAuthority32522.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority32499.actual selector witness) * (LeftAuthority32522.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32526

namespace LeftBound32534
def owner : Owner := ⟨.program ⟨214⟩, ⟨17733⟩⟩
def transferEvent : Nat := 32534
def frameStart : Nat := 32438
def rule : BoundRule := .sum [.predecessor 0 32532 .coefficient, .predecessor 1 32533 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32532 .coefficient)
      LeftAuthority32530.bound (LeftAuthority32530.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32531RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32530.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32530.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32533 .coefficient)
      LeftBound32526.bound (LeftBound32526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32526.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority32530.bound, LeftBound32526.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32530.bound, LeftBound32526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority32530.actual selector witness, LeftBound32526.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32534

namespace LeftBound32538
def owner : Owner := ⟨.program ⟨214⟩, ⟨29423⟩⟩
def transferEvent : Nat := 32538
def frameStart : Nat := 32438
def rule : BoundRule := .sum [.predecessor 0 32536 .coefficient, .predecessor 1 32537 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32536 .coefficient)
      LeftBound32534.bound (LeftBound32534.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32535RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32534.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32534.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32537 .coefficient)
      LeftBound32515.bound (LeftBound32515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32515.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound32534.bound, LeftBound32515.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32534.bound, LeftBound32515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound32534.actual selector witness, LeftBound32515.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32538

namespace LeftBound32551
def owner : Owner := ⟨.program ⟨214⟩, ⟨29420⟩⟩
def transferEvent : Nat := 32551
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 32549 .coefficient, .predecessor 1 32550 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32549 .coefficient)
      LeftBound32380.bound (LeftBound32380.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32380.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32380.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32550 .coefficient)
      LeftBound32363.bound (LeftBound32363.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32363.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32363.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound32380.bound, LeftBound32363.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32380.bound, LeftBound32363.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound32380.actual selector witness, LeftBound32363.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32551

namespace LeftBound32554
def owner : Owner := ⟨.program ⟨214⟩, ⟨29420⟩⟩
def transferEvent : Nat := 32554
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 32548 .summary, .result 32370 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 32548 .summary)
      LeftBound32382.bound (LeftBound32382.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22351⟩⟩) (rawTerms := some (Proof.Events127.exact32548RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32382.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 32370 .summary)
      LeftBound32365.bound (LeftBound32365.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29419⟩⟩) (rawTerms := some (Proof.Events126.exact32370RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32365.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound32382.bound, LeftBound32365.bound]
def bound : CoeffClass := .finite ⟨1292382248169874534400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32382.bound, LeftBound32365.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound32382.actual selector witness, LeftBound32365.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32554

namespace LeftBound32558
def owner : Owner := ⟨.program ⟨214⟩, ⟨29421⟩⟩
def transferEvent : Nat := 32558
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 32556 .coefficient) (.predecessor 1 32557 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32556 .coefficient)
      LeftBound32551.bound (LeftBound32551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32555RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32551.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32551.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32557 .coefficient)
      LeftBound5578.bound (LeftBound5578.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5579RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5578.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5578.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound32551.bound LeftBound5578.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32551.bound, LeftBound5578.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound32551.actual selector witness) * (LeftBound5578.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32558

namespace LeftBound32559
def owner : Owner := ⟨.program ⟨214⟩, ⟨29421⟩⟩
def transferEvent : Nat := 32559
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6665⟩⟩]⟩ [⟨.result 5575 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5575 .coefficient)
      LeftAuthority5574.bound (LeftAuthority5574.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6665⟩⟩) (rawTerms := some (Proof.Events021.exact5575RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5574.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5574.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5574.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5574.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5574.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound32559

namespace LeftBound32560
def owner : Owner := ⟨.program ⟨214⟩, ⟨29421⟩⟩
def transferEvent : Nat := 32560
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 32555 .summary) (.transfer 32559) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 32555 .summary)
      LeftBound32554.bound (LeftBound32554.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29420⟩⟩) (rawTerms := some (Proof.Events127.exact32555RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32554.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 32559)
      LeftBound32559.bound (LeftBound32559.actual selector witness) := by
  exact .transfer (LeftBound32559.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound32554.bound LeftBound32559.bound
def bound : CoeffClass := .finite ⟨4743063528899410259240550400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32554.bound, LeftBound32559.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound32554.actual selector witness) * (LeftBound32559.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32560

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
