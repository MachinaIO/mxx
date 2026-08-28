import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard040
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard041
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard650

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound95662
def owner : Owner := ⟨.program ⟨214⟩, ⟨29570⟩⟩
def transferEvent : Nat := 95662
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 95656 .summary, .result 95502 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95656 .summary)
      LeftBound95514.bound (LeftBound95514.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22544⟩⟩) (rawTerms := some (Proof.Events373.exact95656RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95514.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95502 .summary)
      LeftBound95497.bound (LeftBound95497.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29569⟩⟩) (rawTerms := some (Proof.Events373.exact95502RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95497.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95514.bound, LeftBound95497.bound]
def bound : CoeffClass := .finite ⟨1292449485504936292352, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95514.bound, LeftBound95497.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95514.actual selector witness, LeftBound95497.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95662

namespace LeftBound95686
def owner : Owner := ⟨.program ⟨214⟩, ⟨12741⟩⟩
def transferEvent : Nat := 95686
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 95684 .coefficient) (.predecessor 1 95685 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95684 .coefficient)
      LeftAuthority4634.bound (LeftAuthority4634.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4634.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4634.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95685 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4634.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4634.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4634.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound95686

namespace LeftBound95691
def owner : Owner := ⟨.program ⟨214⟩, ⟨7124⟩⟩
def transferEvent : Nat := 95691
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 95689 .coefficient) (.predecessor 1 95690 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95689 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95690 .coefficient)
      LeftBound7974.bound (LeftBound7974.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact7975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7974.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound7974.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound7974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound7974.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95691

namespace LeftBound95696
def owner : Owner := ⟨.program ⟨214⟩, ⟨12742⟩⟩
def transferEvent : Nat := 95696
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 95694 .coefficient, .predecessor 1 95695 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95694 .coefficient)
      LeftBound95691.bound (LeftBound95691.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95691.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95691.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95695 .coefficient)
      LeftBound95686.bound (LeftBound95686.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95688RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95686.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95686.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95691.bound, LeftBound95686.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95691.bound, LeftBound95686.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95691.actual selector witness, LeftBound95686.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95696

namespace LeftBound95700
def owner : Owner := ⟨.program ⟨214⟩, ⟨12743⟩⟩
def transferEvent : Nat := 95700
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 95698 .coefficient, .predecessor 1 95699 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95698 .coefficient)
      LeftBound95696.bound (LeftBound95696.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95697RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95696.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95696.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95699 .coefficient)
      LeftBound7966.bound (LeftBound7966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact7967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7966.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95696.bound, LeftBound7966.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95696.bound, LeftBound7966.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95696.actual selector witness, LeftBound7966.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95700

namespace LeftBound95701
def owner : Owner := ⟨.program ⟨214⟩, ⟨12743⟩⟩
def transferEvent : Nat := 95701
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨101⟩⟩]⟩ [⟨.result 7967 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7967 .coefficient)
      LeftBound7966.bound (LeftBound7966.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨101⟩⟩) (rawTerms := some (Proof.Events031.exact7967RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7966.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound7966.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7966.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound7966.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound95701

namespace LeftBound95706
def owner : Owner := ⟨.program ⟨214⟩, ⟨12744⟩⟩
def transferEvent : Nat := 95706
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 95704 .coefficient) (.predecessor 1 95705 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95704 .coefficient)
      LeftBound95700.bound (LeftBound95700.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95703RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95700.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95700.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95705 .coefficient)
      LeftAuthority4637.bound (LeftAuthority4637.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4638RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4637.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4637.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound95700.bound LeftAuthority4637.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95700.bound, LeftAuthority4637.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound95700.actual selector witness) * (LeftAuthority4637.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95706

namespace LeftBound95707
def owner : Owner := ⟨.program ⟨214⟩, ⟨12744⟩⟩
def transferEvent : Nat := 95707
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨10015⟩⟩], []⟩ [⟨.result 4638 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4638 .coefficient)
      LeftAuthority4637.bound (LeftAuthority4637.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨10015⟩⟩) (rawTerms := some (Proof.Events018.exact4638RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4637.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4637.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4637.bound []
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4637.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority4637.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound95707

namespace LeftBound95708
def owner : Owner := ⟨.program ⟨214⟩, ⟨12744⟩⟩
def transferEvent : Nat := 95708
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 95703 .summary) (.transfer 95707) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95703 .summary)
      LeftBound95701.bound (LeftBound95701.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12743⟩⟩) (rawTerms := some (Proof.Events373.exact95703RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95701.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 95707)
      LeftBound95707.bound (LeftBound95707.actual selector witness) := by
  exact .transfer (LeftBound95707.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound95701.bound LeftBound95707.bound
def bound : CoeffClass := .finite ⟨38272, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95701.bound, LeftBound95707.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound95701.actual selector witness) * (LeftBound95707.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95708

namespace LeftBound95714
def owner : Owner := ⟨.program ⟨214⟩, ⟨10016⟩⟩
def transferEvent : Nat := 95714
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 95712 .coefficient) (.predecessor 1 95713 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95712 .coefficient)
      LeftAuthority4637.bound (LeftAuthority4637.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4638RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4637.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4637.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95713 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4637.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4637.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4637.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound95714

namespace LeftBound95719
def owner : Owner := ⟨.program ⟨214⟩, ⟨7104⟩⟩
def transferEvent : Nat := 95719
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 95717 .coefficient) (.predecessor 1 95718 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95717 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95718 .coefficient)
      LeftBound8015.bound (LeftBound8015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8015.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound8015.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound8015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound8015.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95719

namespace LeftBound95724
def owner : Owner := ⟨.program ⟨214⟩, ⟨10017⟩⟩
def transferEvent : Nat := 95724
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 95722 .coefficient, .predecessor 1 95723 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95722 .coefficient)
      LeftBound95719.bound (LeftBound95719.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95721RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95719.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95719.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95723 .coefficient)
      LeftBound95714.bound (LeftBound95714.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95716RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95714.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95714.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95719.bound, LeftBound95714.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95719.bound, LeftBound95714.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95719.actual selector witness, LeftBound95714.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95724

namespace LeftBound95728
def owner : Owner := ⟨.program ⟨214⟩, ⟨10018⟩⟩
def transferEvent : Nat := 95728
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 95726 .coefficient, .predecessor 1 95727 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95726 .coefficient)
      LeftBound95724.bound (LeftBound95724.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95724.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95724.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95727 .coefficient)
      LeftBound8007.bound (LeftBound8007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8008RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8007.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95724.bound, LeftBound8007.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95724.bound, LeftBound8007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95724.actual selector witness, LeftBound8007.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95728

namespace LeftBound95729
def owner : Owner := ⟨.program ⟨214⟩, ⟨10018⟩⟩
def transferEvent : Nat := 95729
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨81⟩⟩]⟩ [⟨.result 8008 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8008 .coefficient)
      LeftBound8007.bound (LeftBound8007.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨81⟩⟩) (rawTerms := some (Proof.Events031.exact8008RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8007.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound8007.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound8007.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound95729

namespace LeftBound95734
def owner : Owner := ⟨.program ⟨214⟩, ⟨10019⟩⟩
def transferEvent : Nat := 95734
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 95732 .coefficient) (.predecessor 1 95733 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95732 .coefficient)
      LeftBound95728.bound (LeftBound95728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95731RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95728.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95733 .coefficient)
      LeftBound8004.bound (LeftBound8004.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8004.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8004.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95728.bound LeftBound8004.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95728.bound, LeftBound8004.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95728.actual selector witness) * (LeftBound8004.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95734

namespace LeftBound95735
def owner : Owner := ⟨.program ⟨214⟩, ⟨10019⟩⟩
def transferEvent : Nat := 95735
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩ [⟨.result 8001 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8001 .coefficient)
      LeftAuthority8000.bound (LeftAuthority8000.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7873⟩⟩) (rawTerms := some (Proof.Events031.exact8001RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8000.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8000.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority8000.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority8000.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound95735

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
