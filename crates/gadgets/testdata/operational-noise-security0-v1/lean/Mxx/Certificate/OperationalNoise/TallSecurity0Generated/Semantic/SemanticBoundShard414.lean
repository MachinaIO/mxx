import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard355
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard413

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound61727
def owner : Owner := ⟨.program ⟨214⟩, ⟨16638⟩⟩
def transferEvent : Nat := 61727
def frameStart : Nat := 61688
def rule : BoundRule := .identity (.predecessor 0 61726 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61726 .coefficient)
      LeftAuthority61724.bound (LeftAuthority61724.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61724.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61724.derived selector witness)

def rawBound : CoeffClass := LeftAuthority61724.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61724.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority61724.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound61727

namespace LeftBound61744
def owner : Owner := ⟨.program ⟨214⟩, ⟨16712⟩⟩
def transferEvent : Nat := 61744
def frameStart : Nat := 61688
def rule : BoundRule := .sum [.predecessor 0 61742 .coefficient, .predecessor 1 61743 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61742 .coefficient)
      LeftBound61727.bound (LeftBound61727.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound61727.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61743 .coefficient)
      LeftAuthority61740.bound (LeftAuthority61740.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority61740.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61727.bound, LeftAuthority61740.bound]
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61727.bound, LeftAuthority61740.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound61727.actual selector witness, LeftAuthority61740.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61744

namespace LeftBound61747
def owner : Owner := ⟨.program ⟨214⟩, ⟨16713⟩⟩
def transferEvent : Nat := 61747
def frameStart : Nat := 61688
def rule : BoundRule := .identity (.predecessor 0 61746 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61746 .coefficient)
      LeftBound61744.bound (LeftBound61744.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound61744.derived selector witness)

def rawBound : CoeffClass := LeftBound61744.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61744.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound61744.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound61747

namespace LeftBound61753
def owner : Owner := ⟨.program ⟨214⟩, ⟨16714⟩⟩
def transferEvent : Nat := 61753
def frameStart : Nat := 61688
def rule : BoundRule := .product (.predecessor 0 61751 .coefficient) (.predecessor 1 61752 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61751 .coefficient)
      LeftAuthority61749.bound (LeftAuthority61749.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61749.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61749.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61752 .coefficient)
      LeftBound61747.bound (LeftBound61747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61748RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61747.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61747.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority61749.bound LeftBound61747.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61749.bound, LeftBound61747.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority61749.actual selector witness) * (LeftBound61747.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61753

namespace LeftBound61761
def owner : Owner := ⟨.program ⟨214⟩, ⟨16715⟩⟩
def transferEvent : Nat := 61761
def frameStart : Nat := 61688
def rule : BoundRule := .sum [.predecessor 0 61759 .coefficient, .predecessor 1 61760 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61759 .coefficient)
      LeftAuthority61757.bound (LeftAuthority61757.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61758RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61757.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61757.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61760 .coefficient)
      LeftBound61753.bound (LeftBound61753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61755RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61753.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61753.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority61757.bound, LeftBound61753.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61757.bound, LeftBound61753.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority61757.actual selector witness, LeftBound61753.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61761

namespace LeftBound61765
def owner : Owner := ⟨.program ⟨214⟩, ⟨29392⟩⟩
def transferEvent : Nat := 61765
def frameStart : Nat := 61688
def rule : BoundRule := .product (.predecessor 0 61763 .coefficient) (.predecessor 1 61764 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61763 .coefficient)
      LeftBound61761.bound (LeftBound61761.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61761.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61761.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61764 .coefficient)
      LeftAuthority61738.bound (LeftAuthority61738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61739RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61738.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61738.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound61761.bound LeftAuthority61738.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61761.bound, LeftAuthority61738.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound61761.actual selector witness) * (LeftAuthority61738.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61765

namespace LeftBound61776
def owner : Owner := ⟨.program ⟨214⟩, ⟨17724⟩⟩
def transferEvent : Nat := 61776
def frameStart : Nat := 61688
def rule : BoundRule := .product (.predecessor 0 61774 .coefficient) (.predecessor 1 61775 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61774 .coefficient)
      LeftAuthority61749.bound (LeftAuthority61749.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61749.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61749.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61775 .coefficient)
      LeftAuthority61772.bound (LeftAuthority61772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61773RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61772.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61772.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority61749.bound LeftAuthority61772.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61749.bound, LeftAuthority61772.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority61749.actual selector witness) * (LeftAuthority61772.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61776

namespace LeftBound61784
def owner : Owner := ⟨.program ⟨214⟩, ⟨17725⟩⟩
def transferEvent : Nat := 61784
def frameStart : Nat := 61688
def rule : BoundRule := .sum [.predecessor 0 61782 .coefficient, .predecessor 1 61783 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61782 .coefficient)
      LeftAuthority61780.bound (LeftAuthority61780.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61781RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61780.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61780.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61783 .coefficient)
      LeftBound61776.bound (LeftBound61776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61776.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority61780.bound, LeftBound61776.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61780.bound, LeftBound61776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority61780.actual selector witness, LeftBound61776.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61784

namespace LeftBound61788
def owner : Owner := ⟨.program ⟨214⟩, ⟨29397⟩⟩
def transferEvent : Nat := 61788
def frameStart : Nat := 61688
def rule : BoundRule := .sum [.predecessor 0 61786 .coefficient, .predecessor 1 61787 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61786 .coefficient)
      LeftBound61784.bound (LeftBound61784.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61785RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61784.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61784.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61787 .coefficient)
      LeftBound61765.bound (LeftBound61765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61770RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61765.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61765.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61784.bound, LeftBound61765.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61784.bound, LeftBound61765.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound61784.actual selector witness, LeftBound61765.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61788

namespace LeftBound61801
def owner : Owner := ⟨.program ⟨214⟩, ⟨29394⟩⟩
def transferEvent : Nat := 61801
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 61799 .coefficient, .predecessor 1 61800 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61799 .coefficient)
      LeftBound61630.bound (LeftBound61630.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61798RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61630.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61630.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61800 .coefficient)
      LeftBound61613.bound (LeftBound61613.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events240.exact61620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61613.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61613.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61630.bound, LeftBound61613.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61630.bound, LeftBound61613.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound61630.actual selector witness, LeftBound61613.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61801

namespace LeftBound61804
def owner : Owner := ⟨.program ⟨214⟩, ⟨29394⟩⟩
def transferEvent : Nat := 61804
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 61798 .summary, .result 61620 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61798 .summary)
      LeftBound61632.bound (LeftBound61632.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22343⟩⟩) (rawTerms := some (Proof.Events241.exact61798RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61632.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61620 .summary)
      LeftBound61615.bound (LeftBound61615.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29393⟩⟩) (rawTerms := some (Proof.Events240.exact61620RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61615.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61632.bound, LeftBound61615.bound]
def bound : CoeffClass := .finite ⟨1292382248169874534400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61632.bound, LeftBound61615.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound61632.actual selector witness, LeftBound61615.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61804

namespace LeftBound61808
def owner : Owner := ⟨.program ⟨214⟩, ⟨29395⟩⟩
def transferEvent : Nat := 61808
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 61806 .coefficient) (.predecessor 1 61807 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61806 .coefficient)
      LeftBound61801.bound (LeftBound61801.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61805RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61801.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61801.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61807 .coefficient)
      LeftBound5578.bound (LeftBound5578.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5579RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5578.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5578.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound61801.bound LeftBound5578.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61801.bound, LeftBound5578.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound61801.actual selector witness) * (LeftBound5578.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61808

namespace LeftBound61809
def owner : Owner := ⟨.program ⟨214⟩, ⟨29395⟩⟩
def transferEvent : Nat := 61809
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6665⟩⟩]⟩ [⟨.result 5575 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5575 .coefficient)
      LeftAuthority5574.bound (LeftAuthority5574.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6665⟩⟩) (rawTerms := some (Proof.Events021.exact5575RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5574.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5574.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5574.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5574.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5574.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound61809

namespace LeftBound61810
def owner : Owner := ⟨.program ⟨214⟩, ⟨29395⟩⟩
def transferEvent : Nat := 61810
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 61805 .summary) (.transfer 61809) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61805 .summary)
      LeftBound61804.bound (LeftBound61804.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29394⟩⟩) (rawTerms := some (Proof.Events241.exact61805RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61804.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 61809)
      LeftBound61809.bound (LeftBound61809.actual selector witness) := by
  exact .transfer (LeftBound61809.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound61804.bound LeftBound61809.bound
def bound : CoeffClass := .finite ⟨4743063528899410259240550400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61804.bound, LeftBound61809.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound61804.actual selector witness) * (LeftBound61809.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61810

namespace LeftBound61825
def owner : Owner := ⟨.program ⟨214⟩, ⟨29176⟩⟩
def transferEvent : Nat := 61825
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 61823 .coefficient) (.predecessor 1 61824 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61823 .coefficient)
      LeftBound52872.bound (LeftBound52872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52872.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61824 .coefficient)
      LeftAuthority61821.bound (LeftAuthority61821.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61822RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61821.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61821.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound52872.bound LeftAuthority61821.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52872.bound, LeftAuthority61821.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound52872.actual selector witness) * (LeftAuthority61821.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61825

namespace LeftBound61826
def owner : Owner := ⟨.program ⟨214⟩, ⟨29176⟩⟩
def transferEvent : Nat := 61826
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29174⟩⟩]⟩ [⟨.result 61822 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61822 .coefficient)
      LeftAuthority61821.bound (LeftAuthority61821.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29174⟩⟩) (rawTerms := some (Proof.Events241.exact61822RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61821.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61821.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority61821.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61821.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority61821.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound61826

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
