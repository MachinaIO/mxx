import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard095
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard196

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound29547
def owner : Owner := ⟨.program ⟨214⟩, ⟨15006⟩⟩
def transferEvent : Nat := 29547
def frameStart : Nat := 29488
def rule : BoundRule := .identity (.predecessor 0 29546 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29546 .coefficient)
      LeftBound29544.bound (LeftBound29544.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound29544.derived selector witness)

def rawBound : CoeffClass := LeftBound29544.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29544.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound29544.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound29547

namespace LeftBound29553
def owner : Owner := ⟨.program ⟨214⟩, ⟨15007⟩⟩
def transferEvent : Nat := 29553
def frameStart : Nat := 29488
def rule : BoundRule := .product (.predecessor 0 29551 .coefficient) (.predecessor 1 29552 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29551 .coefficient)
      LeftAuthority29549.bound (LeftAuthority29549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29549.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29549.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29552 .coefficient)
      LeftBound29547.bound (LeftBound29547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29547.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29547.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority29549.bound LeftBound29547.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29549.bound, LeftBound29547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority29549.actual selector witness) * (LeftBound29547.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29553

namespace LeftBound29561
def owner : Owner := ⟨.program ⟨214⟩, ⟨15008⟩⟩
def transferEvent : Nat := 29561
def frameStart : Nat := 29488
def rule : BoundRule := .sum [.predecessor 0 29559 .coefficient, .predecessor 1 29560 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29559 .coefficient)
      LeftAuthority29557.bound (LeftAuthority29557.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29558RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29557.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29557.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29560 .coefficient)
      LeftBound29553.bound (LeftBound29553.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29555RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29553.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29553.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority29557.bound, LeftBound29553.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29557.bound, LeftBound29553.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority29557.actual selector witness, LeftBound29553.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29561

namespace LeftBound29565
def owner : Owner := ⟨.program ⟨214⟩, ⟨26604⟩⟩
def transferEvent : Nat := 29565
def frameStart : Nat := 29488
def rule : BoundRule := .product (.predecessor 0 29563 .coefficient) (.predecessor 1 29564 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29563 .coefficient)
      LeftBound29561.bound (LeftBound29561.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29562RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29561.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29561.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29564 .coefficient)
      LeftAuthority29538.bound (LeftAuthority29538.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29539RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29538.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29538.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29561.bound LeftAuthority29538.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29561.bound, LeftAuthority29538.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29561.actual selector witness) * (LeftAuthority29538.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29565

namespace LeftBound29576
def owner : Owner := ⟨.program ⟨214⟩, ⟨15324⟩⟩
def transferEvent : Nat := 29576
def frameStart : Nat := 29488
def rule : BoundRule := .product (.predecessor 0 29574 .coefficient) (.predecessor 1 29575 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29574 .coefficient)
      LeftAuthority29549.bound (LeftAuthority29549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29549.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29549.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29575 .coefficient)
      LeftAuthority29572.bound (LeftAuthority29572.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29572.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29572.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority29549.bound LeftAuthority29572.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29549.bound, LeftAuthority29572.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority29549.actual selector witness) * (LeftAuthority29572.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29576

namespace LeftBound29584
def owner : Owner := ⟨.program ⟨214⟩, ⟨15325⟩⟩
def transferEvent : Nat := 29584
def frameStart : Nat := 29488
def rule : BoundRule := .sum [.predecessor 0 29582 .coefficient, .predecessor 1 29583 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29582 .coefficient)
      LeftAuthority29580.bound (LeftAuthority29580.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29580.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29580.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29583 .coefficient)
      LeftBound29576.bound (LeftBound29576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29578RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29576.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29576.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority29580.bound, LeftBound29576.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29580.bound, LeftBound29576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority29580.actual selector witness, LeftBound29576.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29584

namespace LeftBound29588
def owner : Owner := ⟨.program ⟨214⟩, ⟨26608⟩⟩
def transferEvent : Nat := 29588
def frameStart : Nat := 29488
def rule : BoundRule := .sum [.predecessor 0 29586 .coefficient, .predecessor 1 29587 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29586 .coefficient)
      LeftBound29584.bound (LeftBound29584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29585RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29584.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29587 .coefficient)
      LeftBound29565.bound (LeftBound29565.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29565.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29565.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29584.bound, LeftBound29565.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29584.bound, LeftBound29565.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29584.actual selector witness, LeftBound29565.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29588

namespace LeftBound29601
def owner : Owner := ⟨.program ⟨214⟩, ⟨26606⟩⟩
def transferEvent : Nat := 29601
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 29599 .coefficient, .predecessor 1 29600 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29599 .coefficient)
      LeftBound29430.bound (LeftBound29430.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29598RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29430.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29430.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29600 .coefficient)
      LeftBound29413.bound (LeftBound29413.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29413.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29413.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29430.bound, LeftBound29413.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29430.bound, LeftBound29413.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29430.actual selector witness, LeftBound29413.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29601

namespace LeftBound29604
def owner : Owner := ⟨.program ⟨214⟩, ⟨26606⟩⟩
def transferEvent : Nat := 29604
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 29598 .summary, .result 29420 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29598 .summary)
      LeftBound29432.bound (LeftBound29432.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20551⟩⟩) (rawTerms := some (Proof.Events115.exact29598RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29432.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29420 .summary)
      LeftBound29415.bound (LeftBound29415.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26605⟩⟩) (rawTerms := some (Proof.Events114.exact29420RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29415.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29432.bound, LeftBound29415.bound]
def bound : CoeffClass := .finite ⟨1291900380601931935744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29432.bound, LeftBound29415.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29432.actual selector witness, LeftBound29415.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29604

namespace LeftBound29628
def owner : Owner := ⟨.program ⟨214⟩, ⟨10507⟩⟩
def transferEvent : Nat := 29628
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 29626 .coefficient) (.predecessor 1 29627 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29626 .coefficient)
      LeftAuthority1232.bound (LeftAuthority1232.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1233RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1232.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1232.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29627 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1232.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1232.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1232.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound29628

namespace LeftBound29633
def owner : Owner := ⟨.program ⟨214⟩, ⟨7342⟩⟩
def transferEvent : Nat := 29633
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 29631 .coefficient) (.predecessor 1 29632 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29631 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29632 .coefficient)
      LeftBound14988.bound (LeftBound14988.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14989RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14988.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14988.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound14988.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound14988.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound14988.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29633

namespace LeftBound29638
def owner : Owner := ⟨.program ⟨214⟩, ⟨10508⟩⟩
def transferEvent : Nat := 29638
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 29636 .coefficient, .predecessor 1 29637 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29636 .coefficient)
      LeftBound29633.bound (LeftBound29633.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29633.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29633.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29637 .coefficient)
      LeftBound29628.bound (LeftBound29628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29630RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29628.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29628.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29633.bound, LeftBound29628.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29633.bound, LeftBound29628.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29633.actual selector witness, LeftBound29628.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29638

namespace LeftBound29642
def owner : Owner := ⟨.program ⟨214⟩, ⟨10509⟩⟩
def transferEvent : Nat := 29642
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 29640 .coefficient, .predecessor 1 29641 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29640 .coefficient)
      LeftBound29638.bound (LeftBound29638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29639RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29638.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29638.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29641 .coefficient)
      LeftBound14980.bound (LeftBound14980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14980.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29638.bound, LeftBound14980.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29638.bound, LeftBound14980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29638.actual selector witness, LeftBound14980.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29642

namespace LeftBound29643
def owner : Owner := ⟨.program ⟨214⟩, ⟨10509⟩⟩
def transferEvent : Nat := 29643
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨86⟩⟩]⟩ [⟨.result 14981 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14981 .coefficient)
      LeftBound14980.bound (LeftBound14980.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨86⟩⟩) (rawTerms := some (Proof.Events058.exact14981RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14980.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound14980.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound14980.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound29643

namespace LeftBound29648
def owner : Owner := ⟨.program ⟨214⟩, ⟨10510⟩⟩
def transferEvent : Nat := 29648
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 29646 .coefficient) (.predecessor 1 29647 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29646 .coefficient)
      LeftBound29642.bound (LeftBound29642.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29645RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29642.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29642.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29647 .coefficient)
      LeftAuthority1235.bound (LeftAuthority1235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1235.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1235.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound29642.bound LeftAuthority1235.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29642.bound, LeftAuthority1235.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound29642.actual selector witness) * (LeftAuthority1235.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29648

namespace LeftBound29649
def owner : Owner := ⟨.program ⟨214⟩, ⟨10510⟩⟩
def transferEvent : Nat := 29649
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9415⟩⟩], []⟩ [⟨.result 1236 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 1236 .coefficient)
      LeftAuthority1235.bound (LeftAuthority1235.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9415⟩⟩) (rawTerms := some (Proof.Events004.exact1236RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1235.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1235.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1235.bound []
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1235.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority1235.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound29649

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
