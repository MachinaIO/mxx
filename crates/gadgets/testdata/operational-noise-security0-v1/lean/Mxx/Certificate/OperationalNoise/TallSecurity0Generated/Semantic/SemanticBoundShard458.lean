import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard048
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard049
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard457

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound67656
def owner : Owner := ⟨.program ⟨214⟩, ⟨29156⟩⟩
def transferEvent : Nat := 67656
def frameStart : Nat := 67579
def rule : BoundRule := .product (.predecessor 0 67654 .coefficient) (.predecessor 1 67655 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67654 .coefficient)
      LeftBound67652.bound (LeftBound67652.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67652.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67652.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67655 .coefficient)
      LeftAuthority67629.bound (LeftAuthority67629.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67630RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67629.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67629.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound67652.bound LeftAuthority67629.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67652.bound, LeftAuthority67629.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound67652.actual selector witness) * (LeftAuthority67629.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67656

namespace LeftBound67667
def owner : Owner := ⟨.program ⟨214⟩, ⟨18203⟩⟩
def transferEvent : Nat := 67667
def frameStart : Nat := 67579
def rule : BoundRule := .product (.predecessor 0 67665 .coefficient) (.predecessor 1 67666 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67665 .coefficient)
      LeftAuthority67640.bound (LeftAuthority67640.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67641RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67640.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67640.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67666 .coefficient)
      LeftAuthority67663.bound (LeftAuthority67663.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67664RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67663.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67663.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority67640.bound LeftAuthority67663.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67640.bound, LeftAuthority67663.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority67640.actual selector witness) * (LeftAuthority67663.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67667

namespace LeftBound67675
def owner : Owner := ⟨.program ⟨214⟩, ⟨18204⟩⟩
def transferEvent : Nat := 67675
def frameStart : Nat := 67579
def rule : BoundRule := .sum [.predecessor 0 67673 .coefficient, .predecessor 1 67674 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67673 .coefficient)
      LeftAuthority67671.bound (LeftAuthority67671.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67672RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67671.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67671.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67674 .coefficient)
      LeftBound67667.bound (LeftBound67667.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67669RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67667.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67667.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority67671.bound, LeftBound67667.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67671.bound, LeftBound67667.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority67671.actual selector witness, LeftBound67667.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67675

namespace LeftBound67679
def owner : Owner := ⟨.program ⟨214⟩, ⟨29160⟩⟩
def transferEvent : Nat := 67679
def frameStart : Nat := 67579
def rule : BoundRule := .sum [.predecessor 0 67677 .coefficient, .predecessor 1 67678 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67677 .coefficient)
      LeftBound67675.bound (LeftBound67675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67676RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67675.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67675.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67678 .coefficient)
      LeftBound67656.bound (LeftBound67656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67656.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67656.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67675.bound, LeftBound67656.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67675.bound, LeftBound67656.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67675.actual selector witness, LeftBound67656.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67679

namespace LeftBound67692
def owner : Owner := ⟨.program ⟨214⟩, ⟨29158⟩⟩
def transferEvent : Nat := 67692
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 67690 .coefficient, .predecessor 1 67691 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67690 .coefficient)
      LeftBound67521.bound (LeftBound67521.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67521.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67521.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67691 .coefficient)
      LeftBound67504.bound (LeftBound67504.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67511RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67504.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67504.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67521.bound, LeftBound67504.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67521.bound, LeftBound67504.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67521.actual selector witness, LeftBound67504.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67692

namespace LeftBound67695
def owner : Owner := ⟨.program ⟨214⟩, ⟨29158⟩⟩
def transferEvent : Nat := 67695
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 67689 .summary, .result 67511 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67689 .summary)
      LeftBound67523.bound (LeftBound67523.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22263⟩⟩) (rawTerms := some (Proof.Events264.exact67689RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67523.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67511 .summary)
      LeftBound67506.bound (LeftBound67506.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29157⟩⟩) (rawTerms := some (Proof.Events263.exact67511RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67506.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67523.bound, LeftBound67506.bound]
def bound : CoeffClass := .finite ⟨1292337423279833362432, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67523.bound, LeftBound67506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67523.actual selector witness, LeftBound67506.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67695

namespace LeftBound67719
def owner : Owner := ⟨.program ⟨214⟩, ⟨12365⟩⟩
def transferEvent : Nat := 67719
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 67717 .coefficient) (.predecessor 1 67718 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67717 .coefficient)
      LeftAuthority3200.bound (LeftAuthority3200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3200.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3200.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67718 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3200.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3200.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3200.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound67719

namespace LeftBound67724
def owner : Owner := ⟨.program ⟨214⟩, ⟨7203⟩⟩
def transferEvent : Nat := 67724
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 67722 .coefficient) (.predecessor 1 67723 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67722 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67723 .coefficient)
      LeftBound8976.bound (LeftBound8976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8976.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound8976.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound8976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound8976.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67724

namespace LeftBound67729
def owner : Owner := ⟨.program ⟨214⟩, ⟨12366⟩⟩
def transferEvent : Nat := 67729
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 67727 .coefficient, .predecessor 1 67728 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67727 .coefficient)
      LeftBound67724.bound (LeftBound67724.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67726RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67724.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67724.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67728 .coefficient)
      LeftBound67719.bound (LeftBound67719.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67721RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67719.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67719.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67724.bound, LeftBound67719.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67724.bound, LeftBound67719.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67724.actual selector witness, LeftBound67719.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67729

namespace LeftBound67733
def owner : Owner := ⟨.program ⟨214⟩, ⟨12367⟩⟩
def transferEvent : Nat := 67733
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 67731 .coefficient, .predecessor 1 67732 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67731 .coefficient)
      LeftBound67729.bound (LeftBound67729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67730RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67729.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67732 .coefficient)
      LeftBound8968.bound (LeftBound8968.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8969RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8968.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8968.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67729.bound, LeftBound8968.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67729.bound, LeftBound8968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67729.actual selector witness, LeftBound8968.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67733

namespace LeftBound67734
def owner : Owner := ⟨.program ⟨214⟩, ⟨12367⟩⟩
def transferEvent : Nat := 67734
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨99⟩⟩]⟩ [⟨.result 8969 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8969 .coefficient)
      LeftBound8968.bound (LeftBound8968.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨99⟩⟩) (rawTerms := some (Proof.Events035.exact8969RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8968.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8968.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound8968.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound8968.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound67734

namespace LeftBound67739
def owner : Owner := ⟨.program ⟨214⟩, ⟨12368⟩⟩
def transferEvent : Nat := 67739
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 67737 .coefficient) (.predecessor 1 67738 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67737 .coefficient)
      LeftBound67733.bound (LeftBound67733.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67733.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67733.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67738 .coefficient)
      LeftAuthority3203.bound (LeftAuthority3203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3204RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3203.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3203.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound67733.bound LeftAuthority3203.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67733.bound, LeftAuthority3203.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound67733.actual selector witness) * (LeftAuthority3203.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67739

namespace LeftBound67740
def owner : Owner := ⟨.program ⟨214⟩, ⟨12368⟩⟩
def transferEvent : Nat := 67740
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9815⟩⟩], []⟩ [⟨.result 3204 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3204 .coefficient)
      LeftAuthority3203.bound (LeftAuthority3203.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9815⟩⟩) (rawTerms := some (Proof.Events012.exact3204RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3203.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3203.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3203.bound []
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3203.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority3203.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound67740

namespace LeftBound67741
def owner : Owner := ⟨.program ⟨214⟩, ⟨12368⟩⟩
def transferEvent : Nat := 67741
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 67736 .summary) (.transfer 67740) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67736 .summary)
      LeftBound67734.bound (LeftBound67734.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12367⟩⟩) (rawTerms := some (Proof.Events264.exact67736RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67734.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 67740)
      LeftBound67740.bound (LeftBound67740.actual selector witness) := by
  exact .transfer (LeftBound67740.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound67734.bound LeftBound67740.bound
def bound : CoeffClass := .finite ⟨33280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67734.bound, LeftBound67740.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound67734.actual selector witness) * (LeftBound67740.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67741

namespace LeftBound67747
def owner : Owner := ⟨.program ⟨214⟩, ⟨9816⟩⟩
def transferEvent : Nat := 67747
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 67745 .coefficient) (.predecessor 1 67746 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67745 .coefficient)
      LeftAuthority3203.bound (LeftAuthority3203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3204RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3203.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3203.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67746 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3203.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3203.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3203.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound67747

namespace LeftBound67752
def owner : Owner := ⟨.program ⟨214⟩, ⟨7183⟩⟩
def transferEvent : Nat := 67752
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 67750 .coefficient) (.predecessor 1 67751 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67750 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67751 .coefficient)
      LeftBound9017.bound (LeftBound9017.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9018RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9017.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9017.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound9017.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound9017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound9017.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67752

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
