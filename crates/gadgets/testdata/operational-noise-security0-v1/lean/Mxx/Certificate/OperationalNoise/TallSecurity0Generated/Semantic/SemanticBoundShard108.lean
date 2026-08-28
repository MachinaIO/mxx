import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard039
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard107

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound17525
def owner : Owner := ⟨.program ⟨214⟩, ⟨29649⟩⟩
def transferEvent : Nat := 17525
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 7749 .summary) (.transfer 17524) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7749 .summary)
      LeftBound7748.bound (LeftBound7748.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25626⟩⟩) (rawTerms := some (Proof.Events030.exact7749RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound7748.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 17524)
      LeftBound17524.bound (LeftBound17524.actual selector witness) := by
  exact .transfer (LeftBound17524.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound7748.bound LeftBound17524.bound
def bound : CoeffClass := .finite ⟨1292449483693632782336, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7748.bound, LeftBound17524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound7748.actual selector witness) * (LeftBound17524.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17525

namespace LeftBound17536
def owner : Owner := ⟨.program ⟨214⟩, ⟨22498⟩⟩
def transferEvent : Nat := 17536
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 17534 .coefficient) (.value (.predecessor 1 17535 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17534 .coefficient)
      LeftAuthority17532.bound (LeftAuthority17532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17533RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17532.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17532.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17535 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority17532.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17532.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority17532.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound17536

namespace LeftBound17540
def owner : Owner := ⟨.program ⟨214⟩, ⟨22499⟩⟩
def transferEvent : Nat := 17540
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 17538 .coefficient) (.predecessor 1 17539 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17538 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17539 .coefficient)
      LeftBound17536.bound (LeftBound17536.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17537RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17536.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17536.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound17536.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound17536.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound17536.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17540

namespace LeftBound17541
def owner : Owner := ⟨.program ⟨214⟩, ⟨22499⟩⟩
def transferEvent : Nat := 17541
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22496⟩⟩]⟩ [⟨.result 17533 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17533 .coefficient)
      LeftAuthority17532.bound (LeftAuthority17532.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22496⟩⟩) (rawTerms := some (Proof.Events068.exact17533RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17532.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17532.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority17532.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17532.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority17532.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound17541

namespace LeftBound17542
def owner : Owner := ⟨.program ⟨214⟩, ⟨22499⟩⟩
def transferEvent : Nat := 17542
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 17541) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 17541)
      LeftBound17541.bound (LeftBound17541.actual selector witness) := by
  exact .transfer (LeftBound17541.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound17541.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound17541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound17541.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17542

namespace LeftBound17637
def owner : Owner := ⟨.program ⟨214⟩, ⟨16769⟩⟩
def transferEvent : Nat := 17637
def frameStart : Nat := 17598
def rule : BoundRule := .identity (.predecessor 0 17636 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17636 .coefficient)
      LeftAuthority17634.bound (LeftAuthority17634.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17634.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17634.derived selector witness)

def rawBound : CoeffClass := LeftAuthority17634.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17634.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority17634.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound17637

namespace LeftBound17654
def owner : Owner := ⟨.program ⟨214⟩, ⟨16843⟩⟩
def transferEvent : Nat := 17654
def frameStart : Nat := 17598
def rule : BoundRule := .sum [.predecessor 0 17652 .coefficient, .predecessor 1 17653 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17652 .coefficient)
      LeftBound17637.bound (LeftBound17637.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound17637.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17653 .coefficient)
      LeftAuthority17650.bound (LeftAuthority17650.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority17650.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound17637.bound, LeftAuthority17650.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17637.bound, LeftAuthority17650.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound17637.actual selector witness, LeftAuthority17650.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17654

namespace LeftBound17657
def owner : Owner := ⟨.program ⟨214⟩, ⟨16844⟩⟩
def transferEvent : Nat := 17657
def frameStart : Nat := 17598
def rule : BoundRule := .identity (.predecessor 0 17656 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17656 .coefficient)
      LeftBound17654.bound (LeftBound17654.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound17654.derived selector witness)

def rawBound : CoeffClass := LeftBound17654.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17654.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound17654.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound17657

namespace LeftBound17663
def owner : Owner := ⟨.program ⟨214⟩, ⟨16845⟩⟩
def transferEvent : Nat := 17663
def frameStart : Nat := 17598
def rule : BoundRule := .product (.predecessor 0 17661 .coefficient) (.predecessor 1 17662 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17661 .coefficient)
      LeftAuthority17659.bound (LeftAuthority17659.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17660RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17659.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17659.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17662 .coefficient)
      LeftBound17657.bound (LeftBound17657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17658RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17657.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17657.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority17659.bound LeftBound17657.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17659.bound, LeftBound17657.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority17659.actual selector witness) * (LeftBound17657.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17663

namespace LeftBound17671
def owner : Owner := ⟨.program ⟨214⟩, ⟨16846⟩⟩
def transferEvent : Nat := 17671
def frameStart : Nat := 17598
def rule : BoundRule := .sum [.predecessor 0 17669 .coefficient, .predecessor 1 17670 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17669 .coefficient)
      LeftAuthority17667.bound (LeftAuthority17667.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17668RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17667.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17667.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17670 .coefficient)
      LeftBound17663.bound (LeftBound17663.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17663.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17663.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority17667.bound, LeftBound17663.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17667.bound, LeftBound17663.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority17667.actual selector witness, LeftBound17663.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17671

namespace LeftBound17675
def owner : Owner := ⟨.program ⟨214⟩, ⟨29648⟩⟩
def transferEvent : Nat := 17675
def frameStart : Nat := 17598
def rule : BoundRule := .product (.predecessor 0 17673 .coefficient) (.predecessor 1 17674 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17673 .coefficient)
      LeftBound17671.bound (LeftBound17671.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17672RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17671.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17671.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17674 .coefficient)
      LeftAuthority17648.bound (LeftAuthority17648.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17649RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17648.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17648.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound17671.bound LeftAuthority17648.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17671.bound, LeftAuthority17648.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound17671.actual selector witness) * (LeftAuthority17648.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17675

namespace LeftBound17686
def owner : Owner := ⟨.program ⟨214⟩, ⟨17512⟩⟩
def transferEvent : Nat := 17686
def frameStart : Nat := 17598
def rule : BoundRule := .product (.predecessor 0 17684 .coefficient) (.predecessor 1 17685 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17684 .coefficient)
      LeftAuthority17659.bound (LeftAuthority17659.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17660RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17659.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17659.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17685 .coefficient)
      LeftAuthority17682.bound (LeftAuthority17682.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17682.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority17659.bound LeftAuthority17682.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17659.bound, LeftAuthority17682.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority17659.actual selector witness) * (LeftAuthority17682.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17686

namespace LeftBound17694
def owner : Owner := ⟨.program ⟨214⟩, ⟨17513⟩⟩
def transferEvent : Nat := 17694
def frameStart : Nat := 17598
def rule : BoundRule := .sum [.predecessor 0 17692 .coefficient, .predecessor 1 17693 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17692 .coefficient)
      LeftAuthority17690.bound (LeftAuthority17690.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17691RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17690.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17690.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17693 .coefficient)
      LeftBound17686.bound (LeftBound17686.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17688RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17686.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17686.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority17690.bound, LeftBound17686.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17690.bound, LeftBound17686.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority17690.actual selector witness, LeftBound17686.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17694

namespace LeftBound17698
def owner : Owner := ⟨.program ⟨214⟩, ⟨29653⟩⟩
def transferEvent : Nat := 17698
def frameStart : Nat := 17598
def rule : BoundRule := .sum [.predecessor 0 17696 .coefficient, .predecessor 1 17697 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17696 .coefficient)
      LeftBound17694.bound (LeftBound17694.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17695RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17694.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17694.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17697 .coefficient)
      LeftBound17675.bound (LeftBound17675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17680RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17675.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17675.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound17694.bound, LeftBound17675.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17694.bound, LeftBound17675.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound17694.actual selector witness, LeftBound17675.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17698

namespace LeftBound17711
def owner : Owner := ⟨.program ⟨214⟩, ⟨29650⟩⟩
def transferEvent : Nat := 17711
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 17709 .coefficient, .predecessor 1 17710 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17709 .coefficient)
      LeftBound17540.bound (LeftBound17540.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17708RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17540.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17540.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17710 .coefficient)
      LeftBound17523.bound (LeftBound17523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17523.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17523.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound17540.bound, LeftBound17523.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17540.bound, LeftBound17523.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound17540.actual selector witness, LeftBound17523.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17711

namespace LeftBound17714
def owner : Owner := ⟨.program ⟨214⟩, ⟨29650⟩⟩
def transferEvent : Nat := 17714
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 17708 .summary, .result 17530 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17708 .summary)
      LeftBound17542.bound (LeftBound17542.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22499⟩⟩) (rawTerms := some (Proof.Events069.exact17708RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17542.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17530 .summary)
      LeftBound17525.bound (LeftBound17525.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29649⟩⟩) (rawTerms := some (Proof.Events068.exact17530RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17525.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound17542.bound, LeftBound17525.bound]
def bound : CoeffClass := .finite ⟨1292449485504936292352, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17542.bound, LeftBound17525.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound17542.actual selector witness, LeftBound17525.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17714

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
