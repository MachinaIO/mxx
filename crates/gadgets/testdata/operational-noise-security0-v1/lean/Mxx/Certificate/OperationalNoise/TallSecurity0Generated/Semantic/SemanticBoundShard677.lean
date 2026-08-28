import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard068
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard069
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard676

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound98757
def owner : Owner := ⟨.program ⟨214⟩, ⟨7096⟩⟩
def transferEvent : Nat := 98757
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 98755 .coefficient) (.predecessor 1 98756 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98755 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98756 .coefficient)
      LeftBound11522.bound (LeftBound11522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11522.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11522.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound11522.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound11522.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound11522.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98757

namespace LeftBound98762
def owner : Owner := ⟨.program ⟨214⟩, ⟨14185⟩⟩
def transferEvent : Nat := 98762
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 98760 .coefficient, .predecessor 1 98761 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98760 .coefficient)
      LeftBound98757.bound (LeftBound98757.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98759RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98757.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98757.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98761 .coefficient)
      LeftBound98752.bound (LeftBound98752.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98754RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98752.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98752.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98757.bound, LeftBound98752.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98757.bound, LeftBound98752.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98757.actual selector witness, LeftBound98752.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98762

namespace LeftBound98766
def owner : Owner := ⟨.program ⟨214⟩, ⟨14186⟩⟩
def transferEvent : Nat := 98766
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 98764 .coefficient, .predecessor 1 98765 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98764 .coefficient)
      LeftBound98762.bound (LeftBound98762.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98763RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98762.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98762.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98765 .coefficient)
      LeftBound11514.bound (LeftBound11514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11514.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98762.bound, LeftBound11514.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98762.bound, LeftBound11514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98762.actual selector witness, LeftBound11514.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98766

namespace LeftBound98767
def owner : Owner := ⟨.program ⟨214⟩, ⟨14186⟩⟩
def transferEvent : Nat := 98767
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨73⟩⟩]⟩ [⟨.result 11515 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11515 .coefficient)
      LeftBound11514.bound (LeftBound11514.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨73⟩⟩) (rawTerms := some (Proof.Events044.exact11515RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11514.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound11514.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound11514.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound98767

namespace LeftBound98772
def owner : Owner := ⟨.program ⟨214⟩, ⟨14187⟩⟩
def transferEvent : Nat := 98772
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 98770 .coefficient) (.predecessor 1 98771 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98770 .coefficient)
      LeftBound98766.bound (LeftBound98766.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98769RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98766.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98766.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98771 .coefficient)
      LeftBound11511.bound (LeftBound11511.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11511.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11511.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98766.bound LeftBound11511.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98766.bound, LeftBound11511.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98766.actual selector witness) * (LeftBound11511.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98772

namespace LeftBound98773
def owner : Owner := ⟨.program ⟨214⟩, ⟨14187⟩⟩
def transferEvent : Nat := 98773
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩ [⟨.result 11508 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11508 .coefficient)
      LeftAuthority11507.bound (LeftAuthority11507.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7852⟩⟩) (rawTerms := some (Proof.Events044.exact11508RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11507.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11507.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority11507.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11507.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority11507.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound98773

namespace LeftBound98774
def owner : Owner := ⟨.program ⟨214⟩, ⟨14187⟩⟩
def transferEvent : Nat := 98774
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 98769 .summary) (.transfer 98773) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98769 .summary)
      LeftBound98767.bound (LeftBound98767.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14186⟩⟩) (rawTerms := some (Proof.Events385.exact98769RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98767.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 98773)
      LeftBound98773.bound (LeftBound98773.actual selector witness) := by
  exact .transfer (LeftBound98773.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98767.bound LeftBound98773.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98767.bound, LeftBound98773.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98767.actual selector witness) * (LeftBound98773.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98774

namespace LeftBound98782
def owner : Owner := ⟨.program ⟨214⟩, ⟨14188⟩⟩
def transferEvent : Nat := 98782
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 98780 .coefficient, .predecessor 1 98781 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98780 .coefficient)
      LeftBound98772.bound (LeftBound98772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98779RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98772.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98772.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98781 .coefficient)
      LeftBound98744.bound (LeftBound98744.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98749RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98744.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98744.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98772.bound, LeftBound98744.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98772.bound, LeftBound98744.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98772.actual selector witness, LeftBound98744.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98782

namespace LeftBound98784
def owner : Owner := ⟨.program ⟨214⟩, ⟨14188⟩⟩
def transferEvent : Nat := 98784
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 98779 .summary, .result 98749 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98779 .summary)
      LeftBound98774.bound (LeftBound98774.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14187⟩⟩) (rawTerms := some (Proof.Events385.exact98779RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98774.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98749 .summary)
      LeftBound98746.bound (LeftBound98746.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14183⟩⟩) (rawTerms := some (Proof.Events385.exact98749RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98746.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98774.bound, LeftBound98746.bound]
def bound : CoeffClass := .finite ⟨95435392, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98774.bound, LeftBound98746.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98774.actual selector witness, LeftBound98746.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98784

namespace LeftBound98788
def owner : Owner := ⟨.program ⟨214⟩, ⟨26054⟩⟩
def transferEvent : Nat := 98788
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 98786 .coefficient) (.predecessor 1 98787 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98786 .coefficient)
      LeftBound98782.bound (LeftBound98782.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98785RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98782.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98782.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98787 .coefficient)
      LeftAuthority98720.bound (LeftAuthority98720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98721RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98720.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98720.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98782.bound LeftAuthority98720.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98782.bound, LeftAuthority98720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98782.actual selector witness) * (LeftAuthority98720.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98788

namespace LeftBound98789
def owner : Owner := ⟨.program ⟨214⟩, ⟨26054⟩⟩
def transferEvent : Nat := 98789
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26053⟩⟩]⟩ [⟨.result 98721 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98721 .coefficient)
      LeftAuthority98720.bound (LeftAuthority98720.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26053⟩⟩) (rawTerms := some (Proof.Events385.exact98721RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98720.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98720.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority98720.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority98720.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound98789

namespace LeftBound98790
def owner : Owner := ⟨.program ⟨214⟩, ⟨26054⟩⟩
def transferEvent : Nat := 98790
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 98785 .summary) (.transfer 98789) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98785 .summary)
      LeftBound98784.bound (LeftBound98784.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14188⟩⟩) (rawTerms := some (Proof.Events385.exact98785RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98784.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 98789)
      LeftBound98789.bound (LeftBound98789.actual selector witness) := by
  exact .transfer (LeftBound98789.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98784.bound LeftBound98789.bound
def bound : CoeffClass := .finite ⟨350249415606272, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98784.bound, LeftBound98789.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98784.actual selector witness) * (LeftBound98789.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98790

namespace LeftBound98801
def owner : Owner := ⟨.program ⟨214⟩, ⟨19519⟩⟩
def transferEvent : Nat := 98801
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 98799 .coefficient) (.value (.predecessor 1 98800 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98799 .coefficient)
      LeftAuthority98797.bound (LeftAuthority98797.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98798RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98797.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98797.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98800 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority98797.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98797.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority98797.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound98801

namespace LeftBound98805
def owner : Owner := ⟨.program ⟨214⟩, ⟨19520⟩⟩
def transferEvent : Nat := 98805
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 98803 .coefficient) (.predecessor 1 98804 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98803 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98804 .coefficient)
      LeftBound98801.bound (LeftBound98801.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98802RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98801.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98801.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound98801.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound98801.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound98801.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98805

namespace LeftBound98806
def owner : Owner := ⟨.program ⟨214⟩, ⟨19520⟩⟩
def transferEvent : Nat := 98806
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19517⟩⟩]⟩ [⟨.result 98798 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98798 .coefficient)
      LeftAuthority98797.bound (LeftAuthority98797.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19517⟩⟩) (rawTerms := some (Proof.Events385.exact98798RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98797.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98797.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority98797.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98797.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority98797.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound98806

namespace LeftBound98807
def owner : Owner := ⟨.program ⟨214⟩, ⟨19520⟩⟩
def transferEvent : Nat := 98807
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 98806) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 98806)
      LeftBound98806.bound (LeftBound98806.actual selector witness) := by
  exact .transfer (LeftBound98806.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound98806.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound98806.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound98806.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98807

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
