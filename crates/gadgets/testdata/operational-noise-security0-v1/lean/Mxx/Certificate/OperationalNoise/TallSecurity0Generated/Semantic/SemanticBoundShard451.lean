import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard040
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard041
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard450

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound66728
def owner : Owner := ⟨.program ⟨214⟩, ⟨29592⟩⟩
def transferEvent : Nat := 66728
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 66726 .coefficient, .predecessor 1 66727 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66726 .coefficient)
      LeftBound66557.bound (LeftBound66557.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66557.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66557.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66727 .coefficient)
      LeftBound66540.bound (LeftBound66540.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66547RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66540.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66540.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66557.bound, LeftBound66540.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66557.bound, LeftBound66540.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66557.actual selector witness, LeftBound66540.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66728

namespace LeftBound66731
def owner : Owner := ⟨.program ⟨214⟩, ⟨29592⟩⟩
def transferEvent : Nat := 66731
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 66725 .summary, .result 66547 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66725 .summary)
      LeftBound66559.bound (LeftBound66559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22551⟩⟩) (rawTerms := some (Proof.Events260.exact66725RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66547 .summary)
      LeftBound66542.bound (LeftBound66542.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29591⟩⟩) (rawTerms := some (Proof.Events259.exact66547RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66542.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66559.bound, LeftBound66542.bound]
def bound : CoeffClass := .finite ⟨1292449485504936292352, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66559.bound, LeftBound66542.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66559.actual selector witness, LeftBound66542.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66731

namespace LeftBound66755
def owner : Owner := ⟨.program ⟨214⟩, ⟨12757⟩⟩
def transferEvent : Nat := 66755
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 66753 .coefficient) (.predecessor 1 66754 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66753 .coefficient)
      LeftAuthority3154.bound (LeftAuthority3154.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3154.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3154.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66754 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3154.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3154.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3154.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound66755

namespace LeftBound66760
def owner : Owner := ⟨.program ⟨214⟩, ⟨7205⟩⟩
def transferEvent : Nat := 66760
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 66758 .coefficient) (.predecessor 1 66759 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66758 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66759 .coefficient)
      LeftBound7974.bound (LeftBound7974.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact7975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7974.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound7974.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound7974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound7974.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66760

namespace LeftBound66765
def owner : Owner := ⟨.program ⟨214⟩, ⟨12758⟩⟩
def transferEvent : Nat := 66765
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 66763 .coefficient, .predecessor 1 66764 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66763 .coefficient)
      LeftBound66760.bound (LeftBound66760.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66760.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66764 .coefficient)
      LeftBound66755.bound (LeftBound66755.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66757RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66755.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66755.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66760.bound, LeftBound66755.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66760.bound, LeftBound66755.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66760.actual selector witness, LeftBound66755.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66765

namespace LeftBound66769
def owner : Owner := ⟨.program ⟨214⟩, ⟨12759⟩⟩
def transferEvent : Nat := 66769
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 66767 .coefficient, .predecessor 1 66768 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66767 .coefficient)
      LeftBound66765.bound (LeftBound66765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66766RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66765.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66765.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66768 .coefficient)
      LeftBound7966.bound (LeftBound7966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact7967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7966.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66765.bound, LeftBound7966.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66765.bound, LeftBound7966.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66765.actual selector witness, LeftBound7966.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66769

namespace LeftBound66770
def owner : Owner := ⟨.program ⟨214⟩, ⟨12759⟩⟩
def transferEvent : Nat := 66770
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
end LeftBound66770

namespace LeftBound66775
def owner : Owner := ⟨.program ⟨214⟩, ⟨12760⟩⟩
def transferEvent : Nat := 66775
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 66773 .coefficient) (.predecessor 1 66774 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66773 .coefficient)
      LeftBound66769.bound (LeftBound66769.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66772RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66769.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66769.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66774 .coefficient)
      LeftAuthority3157.bound (LeftAuthority3157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3158RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3157.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3157.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound66769.bound LeftAuthority3157.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66769.bound, LeftAuthority3157.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound66769.actual selector witness) * (LeftAuthority3157.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66775

namespace LeftBound66776
def owner : Owner := ⟨.program ⟨214⟩, ⟨12760⟩⟩
def transferEvent : Nat := 66776
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨10025⟩⟩], []⟩ [⟨.result 3158 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3158 .coefficient)
      LeftAuthority3157.bound (LeftAuthority3157.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨10025⟩⟩) (rawTerms := some (Proof.Events012.exact3158RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3157.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3157.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3157.bound []
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3157.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority3157.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound66776

namespace LeftBound66777
def owner : Owner := ⟨.program ⟨214⟩, ⟨12760⟩⟩
def transferEvent : Nat := 66777
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 66772 .summary) (.transfer 66776) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66772 .summary)
      LeftBound66770.bound (LeftBound66770.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12759⟩⟩) (rawTerms := some (Proof.Events260.exact66772RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66770.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 66776)
      LeftBound66776.bound (LeftBound66776.actual selector witness) := by
  exact .transfer (LeftBound66776.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound66770.bound LeftBound66776.bound
def bound : CoeffClass := .finite ⟨38272, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66770.bound, LeftBound66776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound66770.actual selector witness) * (LeftBound66776.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66777

namespace LeftBound66783
def owner : Owner := ⟨.program ⟨214⟩, ⟨10026⟩⟩
def transferEvent : Nat := 66783
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 66781 .coefficient) (.predecessor 1 66782 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66781 .coefficient)
      LeftAuthority3157.bound (LeftAuthority3157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3158RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3157.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3157.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66782 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3157.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3157.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3157.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound66783

namespace LeftBound66788
def owner : Owner := ⟨.program ⟨214⟩, ⟨7185⟩⟩
def transferEvent : Nat := 66788
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 66786 .coefficient) (.predecessor 1 66787 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66786 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66787 .coefficient)
      LeftBound8015.bound (LeftBound8015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8015.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound8015.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound8015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound8015.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66788

namespace LeftBound66793
def owner : Owner := ⟨.program ⟨214⟩, ⟨10027⟩⟩
def transferEvent : Nat := 66793
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 66791 .coefficient, .predecessor 1 66792 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66791 .coefficient)
      LeftBound66788.bound (LeftBound66788.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66788.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66788.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66792 .coefficient)
      LeftBound66783.bound (LeftBound66783.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66785RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66783.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66783.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66788.bound, LeftBound66783.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66788.bound, LeftBound66783.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66788.actual selector witness, LeftBound66783.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66793

namespace LeftBound66797
def owner : Owner := ⟨.program ⟨214⟩, ⟨10028⟩⟩
def transferEvent : Nat := 66797
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 66795 .coefficient, .predecessor 1 66796 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66795 .coefficient)
      LeftBound66793.bound (LeftBound66793.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66794RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66793.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66793.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66796 .coefficient)
      LeftBound8007.bound (LeftBound8007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8008RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8007.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66793.bound, LeftBound8007.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66793.bound, LeftBound8007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66793.actual selector witness, LeftBound8007.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66797

namespace LeftBound66798
def owner : Owner := ⟨.program ⟨214⟩, ⟨10028⟩⟩
def transferEvent : Nat := 66798
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
end LeftBound66798

namespace LeftBound66803
def owner : Owner := ⟨.program ⟨214⟩, ⟨10029⟩⟩
def transferEvent : Nat := 66803
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 66801 .coefficient) (.predecessor 1 66802 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66801 .coefficient)
      LeftBound66797.bound (LeftBound66797.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66800RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66797.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66797.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66802 .coefficient)
      LeftBound8004.bound (LeftBound8004.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8004.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8004.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66797.bound LeftBound8004.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66797.bound, LeftBound8004.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66797.actual selector witness) * (LeftBound8004.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66803

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
