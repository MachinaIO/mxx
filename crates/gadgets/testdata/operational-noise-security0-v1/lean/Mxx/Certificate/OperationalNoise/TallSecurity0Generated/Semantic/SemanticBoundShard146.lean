import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard040
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard145

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound22683
def owner : Owner := ⟨.program ⟨214⟩, ⟨22567⟩⟩
def transferEvent : Nat := 22683
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22564⟩⟩]⟩ [⟨.result 22675 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22675 .coefficient)
      LeftAuthority22674.bound (LeftAuthority22674.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22564⟩⟩) (rawTerms := some (Proof.Events088.exact22675RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22674.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22674.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority22674.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority22674.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound22683

namespace LeftBound22684
def owner : Owner := ⟨.program ⟨214⟩, ⟨22567⟩⟩
def transferEvent : Nat := 22684
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 22683) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 22683)
      LeftBound22683.bound (LeftBound22683.actual selector witness) := by
  exact .transfer (LeftBound22683.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound22683.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound22683.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound22683.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22684

namespace LeftBound22779
def owner : Owner := ⟨.program ⟨214⟩, ⟨16765⟩⟩
def transferEvent : Nat := 22779
def frameStart : Nat := 22740
def rule : BoundRule := .identity (.predecessor 0 22778 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22778 .coefficient)
      LeftAuthority22776.bound (LeftAuthority22776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22777RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22776.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22776.derived selector witness)

def rawBound : CoeffClass := LeftAuthority22776.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority22776.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound22779

namespace LeftBound22796
def owner : Owner := ⟨.program ⟨214⟩, ⟨16839⟩⟩
def transferEvent : Nat := 22796
def frameStart : Nat := 22740
def rule : BoundRule := .sum [.predecessor 0 22794 .coefficient, .predecessor 1 22795 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22794 .coefficient)
      LeftBound22779.bound (LeftBound22779.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound22779.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22795 .coefficient)
      LeftAuthority22792.bound (LeftAuthority22792.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority22792.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22779.bound, LeftAuthority22792.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22779.bound, LeftAuthority22792.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22779.actual selector witness, LeftAuthority22792.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22796

namespace LeftBound22799
def owner : Owner := ⟨.program ⟨214⟩, ⟨16840⟩⟩
def transferEvent : Nat := 22799
def frameStart : Nat := 22740
def rule : BoundRule := .identity (.predecessor 0 22798 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22798 .coefficient)
      LeftBound22796.bound (LeftBound22796.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound22796.derived selector witness)

def rawBound : CoeffClass := LeftBound22796.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22796.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound22796.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound22799

namespace LeftBound22805
def owner : Owner := ⟨.program ⟨214⟩, ⟨16841⟩⟩
def transferEvent : Nat := 22805
def frameStart : Nat := 22740
def rule : BoundRule := .product (.predecessor 0 22803 .coefficient) (.predecessor 1 22804 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22803 .coefficient)
      LeftAuthority22801.bound (LeftAuthority22801.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22802RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22801.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22801.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22804 .coefficient)
      LeftBound22799.bound (LeftBound22799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22800RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22799.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22799.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority22801.bound LeftBound22799.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22801.bound, LeftBound22799.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority22801.actual selector witness) * (LeftBound22799.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22805

namespace LeftBound22813
def owner : Owner := ⟨.program ⟨214⟩, ⟨16842⟩⟩
def transferEvent : Nat := 22813
def frameStart : Nat := 22740
def rule : BoundRule := .sum [.predecessor 0 22811 .coefficient, .predecessor 1 22812 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22811 .coefficient)
      LeftAuthority22809.bound (LeftAuthority22809.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22810RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22809.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22809.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22812 .coefficient)
      LeftBound22805.bound (LeftBound22805.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22807RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22805.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22805.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority22809.bound, LeftBound22805.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22809.bound, LeftBound22805.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority22809.actual selector witness, LeftBound22805.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22813

namespace LeftBound22817
def owner : Owner := ⟨.program ⟨214⟩, ⟨29642⟩⟩
def transferEvent : Nat := 22817
def frameStart : Nat := 22740
def rule : BoundRule := .product (.predecessor 0 22815 .coefficient) (.predecessor 1 22816 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22815 .coefficient)
      LeftBound22813.bound (LeftBound22813.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22814RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22813.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22813.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22816 .coefficient)
      LeftAuthority22790.bound (LeftAuthority22790.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22791RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22790.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22790.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound22813.bound LeftAuthority22790.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22813.bound, LeftAuthority22790.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound22813.actual selector witness) * (LeftAuthority22790.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22817

namespace LeftBound22828
def owner : Owner := ⟨.program ⟨214⟩, ⟨16808⟩⟩
def transferEvent : Nat := 22828
def frameStart : Nat := 22740
def rule : BoundRule := .product (.predecessor 0 22826 .coefficient) (.predecessor 1 22827 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22826 .coefficient)
      LeftAuthority22801.bound (LeftAuthority22801.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22802RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22801.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22801.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22827 .coefficient)
      LeftAuthority22824.bound (LeftAuthority22824.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22825RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22824.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22824.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority22801.bound LeftAuthority22824.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22801.bound, LeftAuthority22824.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority22801.actual selector witness) * (LeftAuthority22824.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22828

namespace LeftBound22836
def owner : Owner := ⟨.program ⟨214⟩, ⟨16809⟩⟩
def transferEvent : Nat := 22836
def frameStart : Nat := 22740
def rule : BoundRule := .sum [.predecessor 0 22834 .coefficient, .predecessor 1 22835 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22834 .coefficient)
      LeftAuthority22832.bound (LeftAuthority22832.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22833RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22832.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22832.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22835 .coefficient)
      LeftBound22828.bound (LeftBound22828.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22830RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22828.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22828.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority22832.bound, LeftBound22828.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22832.bound, LeftBound22828.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority22832.actual selector witness, LeftBound22828.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22836

namespace LeftBound22840
def owner : Owner := ⟨.program ⟨214⟩, ⟨29646⟩⟩
def transferEvent : Nat := 22840
def frameStart : Nat := 22740
def rule : BoundRule := .sum [.predecessor 0 22838 .coefficient, .predecessor 1 22839 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22838 .coefficient)
      LeftBound22836.bound (LeftBound22836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22837RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22836.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22836.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22839 .coefficient)
      LeftBound22817.bound (LeftBound22817.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22822RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22817.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22817.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22836.bound, LeftBound22817.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22836.bound, LeftBound22817.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22836.actual selector witness, LeftBound22817.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22840

namespace LeftBound22853
def owner : Owner := ⟨.program ⟨214⟩, ⟨29644⟩⟩
def transferEvent : Nat := 22853
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 22851 .coefficient, .predecessor 1 22852 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22851 .coefficient)
      LeftBound22682.bound (LeftBound22682.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22850RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22682.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22682.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22852 .coefficient)
      LeftBound22665.bound (LeftBound22665.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22672RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22665.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22665.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22682.bound, LeftBound22665.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22682.bound, LeftBound22665.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22682.actual selector witness, LeftBound22665.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22853

namespace LeftBound22856
def owner : Owner := ⟨.program ⟨214⟩, ⟨29644⟩⟩
def transferEvent : Nat := 22856
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 22850 .summary, .result 22672 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22850 .summary)
      LeftBound22684.bound (LeftBound22684.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22567⟩⟩) (rawTerms := some (Proof.Events089.exact22850RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22684.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22672 .summary)
      LeftBound22667.bound (LeftBound22667.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29643⟩⟩) (rawTerms := some (Proof.Events088.exact22672RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22667.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22684.bound, LeftBound22667.bound]
def bound : CoeffClass := .finite ⟨1292449485504936292352, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22684.bound, LeftBound22667.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22684.actual selector witness, LeftBound22667.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22856

namespace LeftBound22880
def owner : Owner := ⟨.program ⟨214⟩, ⟨12789⟩⟩
def transferEvent : Nat := 22880
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 22878 .coefficient) (.predecessor 1 22879 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22878 .coefficient)
      LeftAuthority910.bound (LeftAuthority910.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority910.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority910.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22879 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority910.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority910.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority910.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound22880

namespace LeftBound22885
def owner : Owner := ⟨.program ⟨214⟩, ⟨7357⟩⟩
def transferEvent : Nat := 22885
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 22883 .coefficient) (.predecessor 1 22884 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22883 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22884 .coefficient)
      LeftBound7974.bound (LeftBound7974.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact7975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7974.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound7974.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound7974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound7974.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22885

namespace LeftBound22890
def owner : Owner := ⟨.program ⟨214⟩, ⟨12790⟩⟩
def transferEvent : Nat := 22890
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 22888 .coefficient, .predecessor 1 22889 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22888 .coefficient)
      LeftBound22885.bound (LeftBound22885.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22887RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22885.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22885.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22889 .coefficient)
      LeftBound22880.bound (LeftBound22880.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22882RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22880.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22880.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22885.bound, LeftBound22880.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22885.bound, LeftBound22880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22885.actual selector witness, LeftBound22880.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22890

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
