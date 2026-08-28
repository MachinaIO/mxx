import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard095
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard700

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound101589
def owner : Owner := ⟨.program ⟨214⟩, ⟨20528⟩⟩
def transferEvent : Nat := 101589
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20525⟩⟩]⟩ [⟨.result 101581 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101581 .coefficient)
      LeftAuthority101580.bound (LeftAuthority101580.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20525⟩⟩) (rawTerms := some (Proof.Events396.exact101581RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101580.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101580.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority101580.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101580.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority101580.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound101589

namespace LeftBound101590
def owner : Owner := ⟨.program ⟨214⟩, ⟨20528⟩⟩
def transferEvent : Nat := 101590
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 101589) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 101589)
      LeftBound101589.bound (LeftBound101589.actual selector witness) := by
  exact .transfer (LeftBound101589.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound101589.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound101589.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound101589.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101590

namespace LeftBound101661
def owner : Owner := ⟨.program ⟨214⟩, ⟨14944⟩⟩
def transferEvent : Nat := 101661
def frameStart : Nat := 101634
def rule : BoundRule := .identity (.predecessor 0 101660 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101660 .coefficient)
      LeftAuthority101658.bound (LeftAuthority101658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101658.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101658.derived selector witness)

def rawBound : CoeffClass := LeftAuthority101658.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority101658.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound101661

namespace LeftBound101678
def owner : Owner := ⟨.program ⟨214⟩, ⟨14985⟩⟩
def transferEvent : Nat := 101678
def frameStart : Nat := 101634
def rule : BoundRule := .sum [.predecessor 0 101676 .coefficient, .predecessor 1 101677 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101676 .coefficient)
      LeftBound101661.bound (LeftBound101661.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound101661.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101677 .coefficient)
      LeftAuthority101674.bound (LeftAuthority101674.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority101674.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound101661.bound, LeftAuthority101674.bound]
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101661.bound, LeftAuthority101674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound101661.actual selector witness, LeftAuthority101674.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101678

namespace LeftBound101681
def owner : Owner := ⟨.program ⟨214⟩, ⟨14986⟩⟩
def transferEvent : Nat := 101681
def frameStart : Nat := 101634
def rule : BoundRule := .identity (.predecessor 0 101680 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101680 .coefficient)
      LeftBound101678.bound (LeftBound101678.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound101678.derived selector witness)

def rawBound : CoeffClass := LeftBound101678.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound101678.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound101681

namespace LeftBound101687
def owner : Owner := ⟨.program ⟨214⟩, ⟨14987⟩⟩
def transferEvent : Nat := 101687
def frameStart : Nat := 101634
def rule : BoundRule := .product (.predecessor 0 101685 .coefficient) (.predecessor 1 101686 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101685 .coefficient)
      LeftAuthority101683.bound (LeftAuthority101683.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101684RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101683.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101683.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101686 .coefficient)
      LeftBound101681.bound (LeftBound101681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101682RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101681.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101681.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority101683.bound LeftBound101681.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101683.bound, LeftBound101681.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority101683.actual selector witness) * (LeftBound101681.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101687

namespace LeftBound101695
def owner : Owner := ⟨.program ⟨214⟩, ⟨14988⟩⟩
def transferEvent : Nat := 101695
def frameStart : Nat := 101634
def rule : BoundRule := .sum [.predecessor 0 101693 .coefficient, .predecessor 1 101694 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101693 .coefficient)
      LeftAuthority101691.bound (LeftAuthority101691.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101692RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101691.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101691.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101694 .coefficient)
      LeftBound101687.bound (LeftBound101687.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101687.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101687.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority101691.bound, LeftBound101687.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101691.bound, LeftBound101687.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority101691.actual selector witness, LeftBound101687.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101695

namespace LeftBound101699
def owner : Owner := ⟨.program ⟨214⟩, ⟨26530⟩⟩
def transferEvent : Nat := 101699
def frameStart : Nat := 101634
def rule : BoundRule := .product (.predecessor 0 101697 .coefficient) (.predecessor 1 101698 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101697 .coefficient)
      LeftBound101695.bound (LeftBound101695.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101696RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101695.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101695.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101698 .coefficient)
      LeftAuthority101672.bound (LeftAuthority101672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101672.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101672.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101695.bound LeftAuthority101672.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101695.bound, LeftAuthority101672.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101695.actual selector witness) * (LeftAuthority101672.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101699

namespace LeftBound101710
def owner : Owner := ⟨.program ⟨214⟩, ⟨15302⟩⟩
def transferEvent : Nat := 101710
def frameStart : Nat := 101634
def rule : BoundRule := .product (.predecessor 0 101708 .coefficient) (.predecessor 1 101709 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101708 .coefficient)
      LeftAuthority101683.bound (LeftAuthority101683.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101684RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101683.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101683.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101709 .coefficient)
      LeftAuthority101706.bound (LeftAuthority101706.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101707RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101706.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101706.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority101683.bound LeftAuthority101706.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101683.bound, LeftAuthority101706.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority101683.actual selector witness) * (LeftAuthority101706.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101710

namespace LeftBound101718
def owner : Owner := ⟨.program ⟨214⟩, ⟨15303⟩⟩
def transferEvent : Nat := 101718
def frameStart : Nat := 101634
def rule : BoundRule := .sum [.predecessor 0 101716 .coefficient, .predecessor 1 101717 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101716 .coefficient)
      LeftAuthority101714.bound (LeftAuthority101714.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101715RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101714.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101714.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101717 .coefficient)
      LeftBound101710.bound (LeftBound101710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101712RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101710.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101710.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority101714.bound, LeftBound101710.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101714.bound, LeftBound101710.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority101714.actual selector witness, LeftBound101710.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101718

namespace LeftBound101722
def owner : Owner := ⟨.program ⟨214⟩, ⟨26534⟩⟩
def transferEvent : Nat := 101722
def frameStart : Nat := 101634
def rule : BoundRule := .sum [.predecessor 0 101720 .coefficient, .predecessor 1 101721 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101720 .coefficient)
      LeftBound101718.bound (LeftBound101718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101718.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101718.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101721 .coefficient)
      LeftBound101699.bound (LeftBound101699.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101704RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101699.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101699.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound101718.bound, LeftBound101699.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101718.bound, LeftBound101699.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound101718.actual selector witness, LeftBound101699.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101722

namespace LeftBound101735
def owner : Owner := ⟨.program ⟨214⟩, ⟨26532⟩⟩
def transferEvent : Nat := 101735
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 101733 .coefficient, .predecessor 1 101734 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101733 .coefficient)
      LeftBound101588.bound (LeftBound101588.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101588.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101588.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101734 .coefficient)
      LeftBound101571.bound (LeftBound101571.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101578RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101571.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101571.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound101588.bound, LeftBound101571.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101588.bound, LeftBound101571.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound101588.actual selector witness, LeftBound101571.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101735

namespace LeftBound101738
def owner : Owner := ⟨.program ⟨214⟩, ⟨26532⟩⟩
def transferEvent : Nat := 101738
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 101732 .summary, .result 101578 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101732 .summary)
      LeftBound101590.bound (LeftBound101590.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20528⟩⟩) (rawTerms := some (Proof.Events397.exact101732RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101590.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101578 .summary)
      LeftBound101573.bound (LeftBound101573.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26531⟩⟩) (rawTerms := some (Proof.Events396.exact101578RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101573.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound101590.bound, LeftBound101573.bound]
def bound : CoeffClass := .finite ⟨1291900380601931935744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101590.bound, LeftBound101573.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound101590.actual selector witness, LeftBound101573.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101738

namespace LeftBound101762
def owner : Owner := ⟨.program ⟨214⟩, ⟨10459⟩⟩
def transferEvent : Nat := 101762
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 101760 .coefficient) (.predecessor 1 101761 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101760 .coefficient)
      LeftAuthority4956.bound (LeftAuthority4956.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact4957RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4956.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101761 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4956.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4956.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4956.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound101762

namespace LeftBound101767
def owner : Owner := ⟨.program ⟨214⟩, ⟨7109⟩⟩
def transferEvent : Nat := 101767
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 101765 .coefficient) (.predecessor 1 101766 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101765 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101766 .coefficient)
      LeftBound14988.bound (LeftBound14988.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14989RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14988.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14988.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound14988.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound14988.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound14988.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101767

namespace LeftBound101772
def owner : Owner := ⟨.program ⟨214⟩, ⟨10460⟩⟩
def transferEvent : Nat := 101772
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 101770 .coefficient, .predecessor 1 101771 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101770 .coefficient)
      LeftBound101767.bound (LeftBound101767.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101769RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101767.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101767.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101771 .coefficient)
      LeftBound101762.bound (LeftBound101762.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101764RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101762.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101762.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound101767.bound, LeftBound101762.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101767.bound, LeftBound101762.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound101767.actual selector witness, LeftBound101762.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101772

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
