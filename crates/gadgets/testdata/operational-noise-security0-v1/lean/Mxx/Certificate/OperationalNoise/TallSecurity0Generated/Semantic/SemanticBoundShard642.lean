import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound94552
def owner : Owner := ⟨.program ⟨214⟩, ⟨13440⟩⟩
def transferEvent : Nat := 94552
def frameStart : Nat := 94505
def rule : BoundRule := .product (.predecessor 0 94550 .coefficient) (.predecessor 1 94551 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94550 .coefficient)
      LeftAuthority94548.bound (LeftAuthority94548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94549RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94548.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94548.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94551 .coefficient)
      LeftBound94546.bound (LeftBound94546.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94547RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94546.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94546.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority94548.bound LeftBound94546.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94548.bound, LeftBound94546.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority94548.actual selector witness) * (LeftBound94546.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94552

namespace LeftBound94568
def owner : Owner := ⟨.program ⟨214⟩, ⟨7883⟩⟩
def transferEvent : Nat := 94568
def frameStart : Nat := 94505
def rule : BoundRule := .scale (.predecessor 0 94566 .coefficient) (.value (.predecessor 1 94567 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94566 .coefficient)
      LeftAuthority94564.bound (LeftAuthority94564.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94565RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94564.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94564.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94567 .coefficient)
      LeftAuthority94555.bound (LeftAuthority94555.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority94555.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority94564.bound LeftAuthority94555.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94564.bound, LeftAuthority94555.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority94564.actual selector witness) * (LeftAuthority94555.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound94568

namespace LeftBound94571
def owner : Owner := ⟨.program ⟨214⟩, ⟨6770⟩⟩
def transferEvent : Nat := 94571
def frameStart : Nat := 94505
def rule : BoundRule := .identity (.predecessor 0 94570 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94570 .coefficient)
      LeftAuthority94558.bound (LeftAuthority94558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94559RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94558.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94558.derived selector witness)

def rawBound : CoeffClass := LeftAuthority94558.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority94558.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound94571

namespace LeftBound94575
def owner : Owner := ⟨.program ⟨214⟩, ⟨7884⟩⟩
def transferEvent : Nat := 94575
def frameStart : Nat := 94505
def rule : BoundRule := .product (.predecessor 0 94573 .coefficient) (.predecessor 1 94574 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94573 .coefficient)
      LeftBound94571.bound (LeftBound94571.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94572RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94571.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94571.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94574 .coefficient)
      LeftBound94568.bound (LeftBound94568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94569RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94568.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94568.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound94571.bound LeftBound94568.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94571.bound, LeftBound94568.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound94571.actual selector witness) * (LeftBound94568.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94575

namespace LeftBound94580
def owner : Owner := ⟨.program ⟨214⟩, ⟨13441⟩⟩
def transferEvent : Nat := 94580
def frameStart : Nat := 94505
def rule : BoundRule := .sum [.predecessor 0 94578 .coefficient, .predecessor 1 94579 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94578 .coefficient)
      LeftBound94575.bound (LeftBound94575.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94577RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94575.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94575.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94579 .coefficient)
      LeftBound94552.bound (LeftBound94552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94554RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94552.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94552.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94575.bound, LeftBound94552.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94575.bound, LeftBound94552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94575.actual selector witness, LeftBound94552.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94580

namespace LeftBound94584
def owner : Owner := ⟨.program ⟨214⟩, ⟨25748⟩⟩
def transferEvent : Nat := 94584
def frameStart : Nat := 94505
def rule : BoundRule := .product (.predecessor 0 94582 .coefficient) (.predecessor 1 94583 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94582 .coefficient)
      LeftBound94580.bound (LeftBound94580.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94580.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94580.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94583 .coefficient)
      LeftAuthority94537.bound (LeftAuthority94537.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94537.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94537.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound94580.bound LeftAuthority94537.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94580.bound, LeftAuthority94537.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound94580.actual selector witness) * (LeftAuthority94537.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94584

namespace LeftBound94595
def owner : Owner := ⟨.program ⟨214⟩, ⟨17003⟩⟩
def transferEvent : Nat := 94595
def frameStart : Nat := 94505
def rule : BoundRule := .product (.predecessor 0 94593 .coefficient) (.predecessor 1 94594 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94593 .coefficient)
      LeftAuthority94548.bound (LeftAuthority94548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94549RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94548.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94548.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94594 .coefficient)
      LeftAuthority94591.bound (LeftAuthority94591.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94591.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94591.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority94548.bound LeftAuthority94591.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94548.bound, LeftAuthority94591.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority94548.actual selector witness) * (LeftAuthority94591.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94595

namespace LeftBound94603
def owner : Owner := ⟨.program ⟨214⟩, ⟨17004⟩⟩
def transferEvent : Nat := 94603
def frameStart : Nat := 94505
def rule : BoundRule := .sum [.predecessor 0 94601 .coefficient, .predecessor 1 94602 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94601 .coefficient)
      LeftAuthority94599.bound (LeftAuthority94599.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94599.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94602 .coefficient)
      LeftBound94595.bound (LeftBound94595.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94597RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94595.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94595.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority94599.bound, LeftBound94595.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94599.bound, LeftBound94595.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority94599.actual selector witness, LeftBound94595.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94603

namespace LeftBound94607
def owner : Owner := ⟨.program ⟨214⟩, ⟨25749⟩⟩
def transferEvent : Nat := 94607
def frameStart : Nat := 94505
def rule : BoundRule := .sum [.predecessor 0 94605 .coefficient, .predecessor 1 94606 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94605 .coefficient)
      LeftBound94603.bound (LeftBound94603.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94604RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94603.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94603.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94606 .coefficient)
      LeftBound94584.bound (LeftBound94584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94584.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94603.bound, LeftBound94584.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94603.bound, LeftBound94584.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94603.actual selector witness, LeftBound94584.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94607

namespace LeftBound94620
def owner : Owner := ⟨.program ⟨214⟩, ⟨25747⟩⟩
def transferEvent : Nat := 94620
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94618 .coefficient, .predecessor 1 94619 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94618 .coefficient)
      LeftBound94465.bound (LeftBound94465.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94617RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94465.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94619 .coefficient)
      LeftBound94437.bound (LeftBound94437.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94444RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94437.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94437.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94465.bound, LeftBound94437.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94465.bound, LeftBound94437.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94465.actual selector witness, LeftBound94437.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94620

namespace LeftBound94623
def owner : Owner := ⟨.program ⟨214⟩, ⟨25747⟩⟩
def transferEvent : Nat := 94623
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94617 .summary, .result 94444 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94617 .summary)
      LeftBound94467.bound (LeftBound94467.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20240⟩⟩) (rawTerms := some (Proof.Events369.exact94617RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94467.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94444 .summary)
      LeftBound94439.bound (LeftBound94439.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25746⟩⟩) (rawTerms := some (Proof.Events368.exact94444RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94439.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94467.bound, LeftBound94439.bound]
def bound : CoeffClass := .finite ⟨352188964155392, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94467.bound, LeftBound94439.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94467.actual selector witness, LeftBound94439.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94623

namespace LeftBound94627
def owner : Owner := ⟨.program ⟨214⟩, ⟨30063⟩⟩
def transferEvent : Nat := 94627
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 94625 .coefficient) (.predecessor 1 94626 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94625 .coefficient)
      LeftBound94620.bound (LeftBound94620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94624RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94620.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94620.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94626 .coefficient)
      LeftAuthority94359.bound (LeftAuthority94359.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94359.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94359.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound94620.bound LeftAuthority94359.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94620.bound, LeftAuthority94359.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound94620.actual selector witness) * (LeftAuthority94359.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94627

namespace LeftBound94628
def owner : Owner := ⟨.program ⟨214⟩, ⟨30063⟩⟩
def transferEvent : Nat := 94628
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨30061⟩⟩]⟩ [⟨.result 94360 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94360 .coefficient)
      LeftAuthority94359.bound (LeftAuthority94359.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨30061⟩⟩) (rawTerms := some (Proof.Events368.exact94360RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94359.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94359.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority94359.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94359.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority94359.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound94628

namespace LeftBound94629
def owner : Owner := ⟨.program ⟨214⟩, ⟨30063⟩⟩
def transferEvent : Nat := 94629
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94624 .summary) (.transfer 94628) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94624 .summary)
      LeftBound94623.bound (LeftBound94623.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25747⟩⟩) (rawTerms := some (Proof.Events369.exact94624RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94623.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 94628)
      LeftBound94628.bound (LeftBound94628.actual selector witness) := by
  exact .transfer (LeftBound94628.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound94623.bound LeftBound94628.bound
def bound : CoeffClass := .finite ⟨1292539133473715126272, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94623.bound, LeftBound94628.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound94623.actual selector witness) * (LeftBound94628.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94629

namespace LeftBound94640
def owner : Owner := ⟨.program ⟨214⟩, ⟨22831⟩⟩
def transferEvent : Nat := 94640
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 94638 .coefficient) (.value (.predecessor 1 94639 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94638 .coefficient)
      LeftAuthority94636.bound (LeftAuthority94636.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94637RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94636.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94636.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94639 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority94636.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94636.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority94636.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound94640

namespace LeftBound94644
def owner : Owner := ⟨.program ⟨214⟩, ⟨22832⟩⟩
def transferEvent : Nat := 94644
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 94642 .coefficient) (.predecessor 1 94643 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94642 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94643 .coefficient)
      LeftBound94640.bound (LeftBound94640.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94641RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94640.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94640.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound94640.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound94640.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound94640.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94644

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
