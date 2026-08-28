import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard136
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound21654
def owner : Owner := ⟨.program ⟨214⟩, ⟨13461⟩⟩
def transferEvent : Nat := 21654
def frameStart : Nat := 21567
def rule : BoundRule := .sum [.predecessor 0 21652 .coefficient, .predecessor 1 21653 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21652 .coefficient)
      LeftBound21649.bound (LeftBound21649.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21651RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21649.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21649.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21653 .coefficient)
      LeftBound21626.bound (LeftBound21626.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21626.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21626.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21649.bound, LeftBound21626.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21649.bound, LeftBound21626.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21649.actual selector witness, LeftBound21626.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21654

namespace LeftBound21658
def owner : Owner := ⟨.program ⟨214⟩, ⟨25776⟩⟩
def transferEvent : Nat := 21658
def frameStart : Nat := 21567
def rule : BoundRule := .product (.predecessor 0 21656 .coefficient) (.predecessor 1 21657 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21656 .coefficient)
      LeftBound21654.bound (LeftBound21654.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21654.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21654.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21657 .coefficient)
      LeftAuthority21611.bound (LeftAuthority21611.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21612RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21611.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21611.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound21654.bound LeftAuthority21611.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21654.bound, LeftAuthority21611.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound21654.actual selector witness) * (LeftAuthority21611.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21658

namespace LeftBound21669
def owner : Owner := ⟨.program ⟨214⟩, ⟨17025⟩⟩
def transferEvent : Nat := 21669
def frameStart : Nat := 21567
def rule : BoundRule := .product (.predecessor 0 21667 .coefficient) (.predecessor 1 21668 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21667 .coefficient)
      LeftAuthority21622.bound (LeftAuthority21622.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21623RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21622.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21622.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21668 .coefficient)
      LeftAuthority21665.bound (LeftAuthority21665.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21666RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21665.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21665.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority21622.bound LeftAuthority21665.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21622.bound, LeftAuthority21665.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority21622.actual selector witness) * (LeftAuthority21665.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21669

namespace LeftBound21677
def owner : Owner := ⟨.program ⟨214⟩, ⟨17026⟩⟩
def transferEvent : Nat := 21677
def frameStart : Nat := 21567
def rule : BoundRule := .sum [.predecessor 0 21675 .coefficient, .predecessor 1 21676 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21675 .coefficient)
      LeftAuthority21673.bound (LeftAuthority21673.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21674RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21673.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21673.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21676 .coefficient)
      LeftBound21669.bound (LeftBound21669.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21671RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21669.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21669.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority21673.bound, LeftBound21669.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21673.bound, LeftBound21669.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority21673.actual selector witness, LeftBound21669.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21677

namespace LeftBound21681
def owner : Owner := ⟨.program ⟨214⟩, ⟨25777⟩⟩
def transferEvent : Nat := 21681
def frameStart : Nat := 21567
def rule : BoundRule := .sum [.predecessor 0 21679 .coefficient, .predecessor 1 21680 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21679 .coefficient)
      LeftBound21677.bound (LeftBound21677.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21678RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21677.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21677.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21680 .coefficient)
      LeftBound21658.bound (LeftBound21658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21658.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21677.bound, LeftBound21658.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21677.bound, LeftBound21658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21677.actual selector witness, LeftBound21658.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21681

namespace LeftBound21694
def owner : Owner := ⟨.program ⟨214⟩, ⟨25775⟩⟩
def transferEvent : Nat := 21694
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21692 .coefficient, .predecessor 1 21693 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21692 .coefficient)
      LeftBound21515.bound (LeftBound21515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21691RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21515.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21693 .coefficient)
      LeftBound21487.bound (LeftBound21487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21487.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21487.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21515.bound, LeftBound21487.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21515.bound, LeftBound21487.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21515.actual selector witness, LeftBound21487.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21694

namespace LeftBound21697
def owner : Owner := ⟨.program ⟨214⟩, ⟨25775⟩⟩
def transferEvent : Nat := 21697
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 21691 .summary, .result 21494 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21691 .summary)
      LeftBound21517.bound (LeftBound21517.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20263⟩⟩) (rawTerms := some (Proof.Events084.exact21691RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21517.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21494 .summary)
      LeftBound21489.bound (LeftBound21489.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25774⟩⟩) (rawTerms := some (Proof.Events083.exact21494RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21489.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21517.bound, LeftBound21489.bound]
def bound : CoeffClass := .finite ⟨352188964155392, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21517.bound, LeftBound21489.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21517.actual selector witness, LeftBound21489.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21697

namespace LeftBound21701
def owner : Owner := ⟨.program ⟨214⟩, ⟨30185⟩⟩
def transferEvent : Nat := 21701
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21699 .coefficient) (.predecessor 1 21700 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21699 .coefficient)
      LeftBound21694.bound (LeftBound21694.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21698RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21694.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21694.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21700 .coefficient)
      LeftAuthority21404.bound (LeftAuthority21404.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21405RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21404.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21404.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound21694.bound LeftAuthority21404.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21694.bound, LeftAuthority21404.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound21694.actual selector witness) * (LeftAuthority21404.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21701

namespace LeftBound21702
def owner : Owner := ⟨.program ⟨214⟩, ⟨30185⟩⟩
def transferEvent : Nat := 21702
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨30183⟩⟩]⟩ [⟨.result 21405 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21405 .coefficient)
      LeftAuthority21404.bound (LeftAuthority21404.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨30183⟩⟩) (rawTerms := some (Proof.Events083.exact21405RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21404.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21404.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority21404.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21404.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority21404.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound21702

namespace LeftBound21703
def owner : Owner := ⟨.program ⟨214⟩, ⟨30185⟩⟩
def transferEvent : Nat := 21703
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21698 .summary) (.transfer 21702) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21698 .summary)
      LeftBound21697.bound (LeftBound21697.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25775⟩⟩) (rawTerms := some (Proof.Events084.exact21698RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21697.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 21702)
      LeftBound21702.bound (LeftBound21702.actual selector witness) := by
  exact .transfer (LeftBound21702.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound21697.bound LeftBound21702.bound
def bound : CoeffClass := .finite ⟨1292539133473715126272, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21697.bound, LeftBound21702.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound21697.actual selector witness) * (LeftBound21702.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21703

namespace LeftBound21714
def owner : Owner := ⟨.program ⟨214⟩, ⟨22854⟩⟩
def transferEvent : Nat := 21714
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 21712 .coefficient) (.value (.predecessor 1 21713 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21712 .coefficient)
      LeftAuthority21710.bound (LeftAuthority21710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21711RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21710.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21710.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21713 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority21710.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21710.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority21710.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound21714

namespace LeftBound21718
def owner : Owner := ⟨.program ⟨214⟩, ⟨22855⟩⟩
def transferEvent : Nat := 21718
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21716 .coefficient) (.predecessor 1 21717 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21716 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21717 .coefficient)
      LeftBound21714.bound (LeftBound21714.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21715RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21714.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21714.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound21714.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound21714.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound21714.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21718

namespace LeftBound21719
def owner : Owner := ⟨.program ⟨214⟩, ⟨22855⟩⟩
def transferEvent : Nat := 21719
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22852⟩⟩]⟩ [⟨.result 21711 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21711 .coefficient)
      LeftAuthority21710.bound (LeftAuthority21710.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22852⟩⟩) (rawTerms := some (Proof.Events084.exact21711RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21710.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21710.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority21710.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21710.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority21710.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound21719

namespace LeftBound21720
def owner : Owner := ⟨.program ⟨214⟩, ⟨22855⟩⟩
def transferEvent : Nat := 21720
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 21719) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 21719)
      LeftBound21719.bound (LeftBound21719.actual selector witness) := by
  exact .transfer (LeftBound21719.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound21719.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound21719.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound21719.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21720

namespace LeftBound21815
def owner : Owner := ⟨.program ⟨214⟩, ⟨17024⟩⟩
def transferEvent : Nat := 21815
def frameStart : Nat := 21776
def rule : BoundRule := .identity (.predecessor 0 21814 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21814 .coefficient)
      LeftAuthority21812.bound (LeftAuthority21812.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21813RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21812.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21812.derived selector witness)

def rawBound : CoeffClass := LeftAuthority21812.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21812.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority21812.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound21815

namespace LeftBound21832
def owner : Owner := ⟨.program ⟨214⟩, ⟨17063⟩⟩
def transferEvent : Nat := 21832
def frameStart : Nat := 21776
def rule : BoundRule := .sum [.predecessor 0 21830 .coefficient, .predecessor 1 21831 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21830 .coefficient)
      LeftBound21815.bound (LeftBound21815.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound21815.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21831 .coefficient)
      LeftAuthority21828.bound (LeftAuthority21828.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority21828.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21815.bound, LeftAuthority21828.bound]
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21815.bound, LeftAuthority21828.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21815.actual selector witness, LeftAuthority21828.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21832

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
