import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard256

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound38552
def owner : Owner := ⟨.program ⟨214⟩, ⟨19899⟩⟩
def transferEvent : Nat := 38552
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 38551) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 38551)
      LeftBound38551.bound (LeftBound38551.actual selector witness) := by
  exact .transfer (LeftBound38551.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound38551.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound38551.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound38551.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38552

namespace LeftBound38631
def owner : Owner := ⟨.program ⟨214⟩, ⟨12387⟩⟩
def transferEvent : Nat := 38631
def frameStart : Nat := 38602
def rule : BoundRule := .product (.predecessor 0 38629 .coefficient) (.predecessor 1 38630 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38629 .coefficient)
      LeftAuthority38627.bound (LeftAuthority38627.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events150.exact38628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38627.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38627.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38630 .coefficient)
      LeftAuthority38624.bound (LeftAuthority38624.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events150.exact38625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38624.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38624.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority38627.bound LeftAuthority38624.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38627.bound, LeftAuthority38624.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority38627.actual selector witness) * (LeftAuthority38624.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38631

namespace LeftBound38635
def owner : Owner := ⟨.program ⟨214⟩, ⟨12388⟩⟩
def transferEvent : Nat := 38635
def frameStart : Nat := 38602
def rule : BoundRule := .identity (.predecessor 0 38634 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38634 .coefficient)
      LeftBound38631.bound (LeftBound38631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events150.exact38633RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38631.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38631.derived selector witness)

def rawBound : CoeffClass := LeftBound38631.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound38631.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound38635

namespace LeftBound38652
def owner : Owner := ⟨.program ⟨214⟩, ⟨12474⟩⟩
def transferEvent : Nat := 38652
def frameStart : Nat := 38602
def rule : BoundRule := .sum [.predecessor 0 38650 .coefficient, .predecessor 1 38651 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38650 .coefficient)
      LeftBound38635.bound (LeftBound38635.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound38635.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38651 .coefficient)
      LeftAuthority38648.bound (LeftAuthority38648.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority38648.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38635.bound, LeftAuthority38648.bound]
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38635.bound, LeftAuthority38648.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38635.actual selector witness, LeftAuthority38648.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38652

namespace LeftBound38655
def owner : Owner := ⟨.program ⟨214⟩, ⟨12475⟩⟩
def transferEvent : Nat := 38655
def frameStart : Nat := 38602
def rule : BoundRule := .identity (.predecessor 0 38654 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38654 .coefficient)
      LeftBound38652.bound (LeftBound38652.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound38652.derived selector witness)

def rawBound : CoeffClass := LeftBound38652.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38652.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound38652.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound38655

namespace LeftBound38661
def owner : Owner := ⟨.program ⟨214⟩, ⟨12476⟩⟩
def transferEvent : Nat := 38661
def frameStart : Nat := 38602
def rule : BoundRule := .product (.predecessor 0 38659 .coefficient) (.predecessor 1 38660 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38659 .coefficient)
      LeftAuthority38657.bound (LeftAuthority38657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38658RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38657.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38657.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38660 .coefficient)
      LeftBound38655.bound (LeftBound38655.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38656RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38655.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38655.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority38657.bound LeftBound38655.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38657.bound, LeftBound38655.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority38657.actual selector witness) * (LeftBound38655.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38661

namespace LeftBound38677
def owner : Owner := ⟨.program ⟨214⟩, ⟨7868⟩⟩
def transferEvent : Nat := 38677
def frameStart : Nat := 38602
def rule : BoundRule := .scale (.predecessor 0 38675 .coefficient) (.value (.predecessor 1 38676 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38675 .coefficient)
      LeftAuthority38673.bound (LeftAuthority38673.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38674RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38673.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38673.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38676 .coefficient)
      LeftAuthority38664.bound (LeftAuthority38664.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority38664.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority38673.bound LeftAuthority38664.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38673.bound, LeftAuthority38664.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority38673.actual selector witness) * (LeftAuthority38664.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound38677

namespace LeftBound38680
def owner : Owner := ⟨.program ⟨214⟩, ⟨6765⟩⟩
def transferEvent : Nat := 38680
def frameStart : Nat := 38602
def rule : BoundRule := .identity (.predecessor 0 38679 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38679 .coefficient)
      LeftAuthority38667.bound (LeftAuthority38667.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38668RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38667.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38667.derived selector witness)

def rawBound : CoeffClass := LeftAuthority38667.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38667.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority38667.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound38680

namespace LeftBound38684
def owner : Owner := ⟨.program ⟨214⟩, ⟨7869⟩⟩
def transferEvent : Nat := 38684
def frameStart : Nat := 38602
def rule : BoundRule := .product (.predecessor 0 38682 .coefficient) (.predecessor 1 38683 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38682 .coefficient)
      LeftBound38680.bound (LeftBound38680.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38681RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38680.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38680.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38683 .coefficient)
      LeftBound38677.bound (LeftBound38677.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38678RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38677.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38677.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound38680.bound LeftBound38677.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38680.bound, LeftBound38677.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound38680.actual selector witness) * (LeftBound38677.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38684

namespace LeftBound38689
def owner : Owner := ⟨.program ⟨214⟩, ⟨12477⟩⟩
def transferEvent : Nat := 38689
def frameStart : Nat := 38602
def rule : BoundRule := .sum [.predecessor 0 38687 .coefficient, .predecessor 1 38688 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38687 .coefficient)
      LeftBound38684.bound (LeftBound38684.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38686RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38684.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38684.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38688 .coefficient)
      LeftBound38661.bound (LeftBound38661.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38661.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38661.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38684.bound, LeftBound38661.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38684.bound, LeftBound38661.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38684.actual selector witness, LeftBound38661.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38689

namespace LeftBound38693
def owner : Owner := ⟨.program ⟨214⟩, ⟨25386⟩⟩
def transferEvent : Nat := 38693
def frameStart : Nat := 38602
def rule : BoundRule := .product (.predecessor 0 38691 .coefficient) (.predecessor 1 38692 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38691 .coefficient)
      LeftBound38689.bound (LeftBound38689.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38690RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38689.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38689.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38692 .coefficient)
      LeftAuthority38646.bound (LeftAuthority38646.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events150.exact38647RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38646.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38646.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound38689.bound LeftAuthority38646.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38689.bound, LeftAuthority38646.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound38689.actual selector witness) * (LeftAuthority38646.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38693

namespace LeftBound38704
def owner : Owner := ⟨.program ⟨214⟩, ⟨16475⟩⟩
def transferEvent : Nat := 38704
def frameStart : Nat := 38602
def rule : BoundRule := .product (.predecessor 0 38702 .coefficient) (.predecessor 1 38703 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38702 .coefficient)
      LeftAuthority38657.bound (LeftAuthority38657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38658RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38657.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38657.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38703 .coefficient)
      LeftAuthority38700.bound (LeftAuthority38700.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38701RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38700.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38700.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority38657.bound LeftAuthority38700.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38657.bound, LeftAuthority38700.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority38657.actual selector witness) * (LeftAuthority38700.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38704

namespace LeftBound38712
def owner : Owner := ⟨.program ⟨214⟩, ⟨16476⟩⟩
def transferEvent : Nat := 38712
def frameStart : Nat := 38602
def rule : BoundRule := .sum [.predecessor 0 38710 .coefficient, .predecessor 1 38711 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38710 .coefficient)
      LeftAuthority38708.bound (LeftAuthority38708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38709RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38708.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38708.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38711 .coefficient)
      LeftBound38704.bound (LeftBound38704.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38706RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38704.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38704.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority38708.bound, LeftBound38704.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38708.bound, LeftBound38704.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority38708.actual selector witness, LeftBound38704.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38712

namespace LeftBound38716
def owner : Owner := ⟨.program ⟨214⟩, ⟨25387⟩⟩
def transferEvent : Nat := 38716
def frameStart : Nat := 38602
def rule : BoundRule := .sum [.predecessor 0 38714 .coefficient, .predecessor 1 38715 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38714 .coefficient)
      LeftBound38712.bound (LeftBound38712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38712.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38715 .coefficient)
      LeftBound38693.bound (LeftBound38693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38698RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38693.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38693.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38712.bound, LeftBound38693.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38712.bound, LeftBound38693.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38712.actual selector witness, LeftBound38693.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38716

namespace LeftBound38729
def owner : Owner := ⟨.program ⟨214⟩, ⟨25385⟩⟩
def transferEvent : Nat := 38729
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 38727 .coefficient, .predecessor 1 38728 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38727 .coefficient)
      LeftBound38550.bound (LeftBound38550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38726RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38550.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38550.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38728 .coefficient)
      LeftBound38533.bound (LeftBound38533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events150.exact38540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38533.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38533.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38550.bound, LeftBound38533.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38550.bound, LeftBound38533.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38550.actual selector witness, LeftBound38533.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38729

namespace LeftBound38732
def owner : Owner := ⟨.program ⟨214⟩, ⟨25385⟩⟩
def transferEvent : Nat := 38732
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 38726 .summary, .result 38540 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38726 .summary)
      LeftBound38552.bound (LeftBound38552.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19899⟩⟩) (rawTerms := some (Proof.Events151.exact38726RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38552.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38540 .summary)
      LeftBound38535.bound (LeftBound38535.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25384⟩⟩) (rawTerms := some (Proof.Events150.exact38540RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38535.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38552.bound, LeftBound38535.bound]
def bound : CoeffClass := .finite ⟨352127895089152, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38552.bound, LeftBound38535.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38552.actual selector witness, LeftBound38535.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38732

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
