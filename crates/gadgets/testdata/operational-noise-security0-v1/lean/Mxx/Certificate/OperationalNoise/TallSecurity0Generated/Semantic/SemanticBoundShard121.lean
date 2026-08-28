import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard078

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound19643
def owner : Owner := ⟨.program ⟨214⟩, ⟨27479⟩⟩
def transferEvent : Nat := 19643
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 19641 .coefficient) (.predecessor 1 19642 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19641 .coefficient)
      LeftBound12755.bound (LeftBound12755.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12759RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12755.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12755.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19642 .coefficient)
      LeftAuthority19639.bound (LeftAuthority19639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19640RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19639.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19639.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound12755.bound LeftAuthority19639.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12755.bound, LeftAuthority19639.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound12755.actual selector witness) * (LeftAuthority19639.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19643

namespace LeftBound19644
def owner : Owner := ⟨.program ⟨214⟩, ⟨27479⟩⟩
def transferEvent : Nat := 19644
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27477⟩⟩]⟩ [⟨.result 19640 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19640 .coefficient)
      LeftAuthority19639.bound (LeftAuthority19639.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27477⟩⟩) (rawTerms := some (Proof.Events076.exact19640RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19639.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19639.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority19639.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19639.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority19639.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound19644

namespace LeftBound19645
def owner : Owner := ⟨.program ⟨214⟩, ⟨27479⟩⟩
def transferEvent : Nat := 19645
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 12759 .summary) (.transfer 19644) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12759 .summary)
      LeftBound12758.bound (LeftBound12758.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25934⟩⟩) (rawTerms := some (Proof.Events049.exact12759RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound12758.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 19644)
      LeftBound19644.bound (LeftBound19644.actual selector witness) := by
  exact .transfer (LeftBound19644.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound12758.bound LeftBound19644.bound
def bound : CoeffClass := .finite ⟨1292001234793221062656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12758.bound, LeftBound19644.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound12758.actual selector witness) * (LeftBound19644.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19645

namespace LeftBound19656
def owner : Owner := ⟨.program ⟨214⟩, ⟨21058⟩⟩
def transferEvent : Nat := 19656
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 19654 .coefficient) (.value (.predecessor 1 19655 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19654 .coefficient)
      LeftAuthority19652.bound (LeftAuthority19652.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19652.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19652.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19655 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority19652.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19652.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority19652.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound19656

namespace LeftBound19660
def owner : Owner := ⟨.program ⟨214⟩, ⟨21059⟩⟩
def transferEvent : Nat := 19660
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 19658 .coefficient) (.predecessor 1 19659 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19658 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19659 .coefficient)
      LeftBound19656.bound (LeftBound19656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19657RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19656.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19656.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound19656.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound19656.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound19656.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19660

namespace LeftBound19661
def owner : Owner := ⟨.program ⟨214⟩, ⟨21059⟩⟩
def transferEvent : Nat := 19661
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21056⟩⟩]⟩ [⟨.result 19653 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19653 .coefficient)
      LeftAuthority19652.bound (LeftAuthority19652.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21056⟩⟩) (rawTerms := some (Proof.Events076.exact19653RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19652.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19652.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority19652.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19652.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority19652.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound19661

namespace LeftBound19662
def owner : Owner := ⟨.program ⟨214⟩, ⟨21059⟩⟩
def transferEvent : Nat := 19662
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 19661) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 19661)
      LeftBound19661.bound (LeftBound19661.actual selector witness) := by
  exact .transfer (LeftBound19661.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound19661.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound19661.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound19661.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19662

namespace LeftBound19757
def owner : Owner := ⟨.program ⟨214⟩, ⟨15719⟩⟩
def transferEvent : Nat := 19757
def frameStart : Nat := 19718
def rule : BoundRule := .identity (.predecessor 0 19756 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19756 .coefficient)
      LeftAuthority19754.bound (LeftAuthority19754.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events077.exact19755RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19754.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19754.derived selector witness)

def rawBound : CoeffClass := LeftAuthority19754.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19754.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority19754.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound19757

namespace LeftBound19774
def owner : Owner := ⟨.program ⟨214⟩, ⟨15793⟩⟩
def transferEvent : Nat := 19774
def frameStart : Nat := 19718
def rule : BoundRule := .sum [.predecessor 0 19772 .coefficient, .predecessor 1 19773 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19772 .coefficient)
      LeftBound19757.bound (LeftBound19757.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound19757.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19773 .coefficient)
      LeftAuthority19770.bound (LeftAuthority19770.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority19770.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound19757.bound, LeftAuthority19770.bound]
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19757.bound, LeftAuthority19770.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound19757.actual selector witness, LeftAuthority19770.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19774

namespace LeftBound19777
def owner : Owner := ⟨.program ⟨214⟩, ⟨15794⟩⟩
def transferEvent : Nat := 19777
def frameStart : Nat := 19718
def rule : BoundRule := .identity (.predecessor 0 19776 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19776 .coefficient)
      LeftBound19774.bound (LeftBound19774.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound19774.derived selector witness)

def rawBound : CoeffClass := LeftBound19774.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19774.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound19774.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound19777

namespace LeftBound19783
def owner : Owner := ⟨.program ⟨214⟩, ⟨15795⟩⟩
def transferEvent : Nat := 19783
def frameStart : Nat := 19718
def rule : BoundRule := .product (.predecessor 0 19781 .coefficient) (.predecessor 1 19782 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19781 .coefficient)
      LeftAuthority19779.bound (LeftAuthority19779.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events077.exact19780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19779.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19779.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19782 .coefficient)
      LeftBound19777.bound (LeftBound19777.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events077.exact19778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19777.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19777.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority19779.bound LeftBound19777.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19779.bound, LeftBound19777.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority19779.actual selector witness) * (LeftBound19777.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19783

namespace LeftBound19791
def owner : Owner := ⟨.program ⟨214⟩, ⟨15796⟩⟩
def transferEvent : Nat := 19791
def frameStart : Nat := 19718
def rule : BoundRule := .sum [.predecessor 0 19789 .coefficient, .predecessor 1 19790 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19789 .coefficient)
      LeftAuthority19787.bound (LeftAuthority19787.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events077.exact19788RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19787.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19787.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19790 .coefficient)
      LeftBound19783.bound (LeftBound19783.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events077.exact19785RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19783.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19783.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority19787.bound, LeftBound19783.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19787.bound, LeftBound19783.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority19787.actual selector witness, LeftBound19783.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19791

namespace LeftBound19795
def owner : Owner := ⟨.program ⟨214⟩, ⟨27478⟩⟩
def transferEvent : Nat := 19795
def frameStart : Nat := 19718
def rule : BoundRule := .product (.predecessor 0 19793 .coefficient) (.predecessor 1 19794 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19793 .coefficient)
      LeftBound19791.bound (LeftBound19791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events077.exact19792RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19791.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19791.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19794 .coefficient)
      LeftAuthority19768.bound (LeftAuthority19768.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events077.exact19769RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19768.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19768.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound19791.bound LeftAuthority19768.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19791.bound, LeftAuthority19768.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound19791.actual selector witness) * (LeftAuthority19768.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19795

namespace LeftBound19806
def owner : Owner := ⟨.program ⟨214⟩, ⟨17456⟩⟩
def transferEvent : Nat := 19806
def frameStart : Nat := 19718
def rule : BoundRule := .product (.predecessor 0 19804 .coefficient) (.predecessor 1 19805 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19804 .coefficient)
      LeftAuthority19779.bound (LeftAuthority19779.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events077.exact19780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19779.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19779.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19805 .coefficient)
      LeftAuthority19802.bound (LeftAuthority19802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events077.exact19803RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19802.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19802.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority19779.bound LeftAuthority19802.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19779.bound, LeftAuthority19802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority19779.actual selector witness) * (LeftAuthority19802.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19806

namespace LeftBound19814
def owner : Owner := ⟨.program ⟨214⟩, ⟨17457⟩⟩
def transferEvent : Nat := 19814
def frameStart : Nat := 19718
def rule : BoundRule := .sum [.predecessor 0 19812 .coefficient, .predecessor 1 19813 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19812 .coefficient)
      LeftAuthority19810.bound (LeftAuthority19810.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events077.exact19811RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19810.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19810.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19813 .coefficient)
      LeftBound19806.bound (LeftBound19806.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events077.exact19808RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19806.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19806.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority19810.bound, LeftBound19806.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19810.bound, LeftBound19806.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority19810.actual selector witness, LeftBound19806.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19814

namespace LeftBound19818
def owner : Owner := ⟨.program ⟨214⟩, ⟨27483⟩⟩
def transferEvent : Nat := 19818
def frameStart : Nat := 19718
def rule : BoundRule := .sum [.predecessor 0 19816 .coefficient, .predecessor 1 19817 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19816 .coefficient)
      LeftBound19814.bound (LeftBound19814.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events077.exact19815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19814.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19814.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19817 .coefficient)
      LeftBound19795.bound (LeftBound19795.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events077.exact19800RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19795.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19795.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound19814.bound, LeftBound19795.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19814.bound, LeftBound19795.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound19814.actual selector witness, LeftBound19795.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19818

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
