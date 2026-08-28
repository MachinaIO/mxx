import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard102

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound16710
def owner : Owner := ⟨.program ⟨214⟩, ⟨18397⟩⟩
def transferEvent : Nat := 16710
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16708 .coefficient, .predecessor 1 16709 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16708 .coefficient)
      LeftBound16706.bound (LeftBound16706.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16707RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16706.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16706.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16709 .coefficient)
      LeftAuthority16359.bound (LeftAuthority16359.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events063.exact16360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16359.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16359.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16706.bound, LeftAuthority16359.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16706.bound, LeftAuthority16359.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16706.actual selector witness, LeftAuthority16359.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16710

namespace LeftBound16714
def owner : Owner := ⟨.program ⟨214⟩, ⟨18398⟩⟩
def transferEvent : Nat := 16714
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16712 .coefficient, .predecessor 1 16713 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16712 .coefficient)
      LeftBound16710.bound (LeftBound16710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16711RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16710.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16710.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16713 .coefficient)
      LeftAuthority16336.bound (LeftAuthority16336.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events063.exact16337RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16336.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16336.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16710.bound, LeftAuthority16336.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16710.bound, LeftAuthority16336.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16710.actual selector witness, LeftAuthority16336.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16714

namespace LeftBound16718
def owner : Owner := ⟨.program ⟨214⟩, ⟨18399⟩⟩
def transferEvent : Nat := 16718
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16716 .coefficient, .predecessor 1 16717 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16716 .coefficient)
      LeftBound16714.bound (LeftBound16714.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16715RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16714.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16714.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16717 .coefficient)
      LeftAuthority16313.bound (LeftAuthority16313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events063.exact16314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16313.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16313.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16714.bound, LeftAuthority16313.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16714.bound, LeftAuthority16313.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16714.actual selector witness, LeftAuthority16313.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16718

namespace LeftBound16722
def owner : Owner := ⟨.program ⟨214⟩, ⟨18400⟩⟩
def transferEvent : Nat := 16722
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16720 .coefficient, .predecessor 1 16721 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16720 .coefficient)
      LeftBound16718.bound (LeftBound16718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16718.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16718.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16721 .coefficient)
      LeftAuthority16290.bound (LeftAuthority16290.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events063.exact16291RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16290.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16290.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16718.bound, LeftAuthority16290.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16718.bound, LeftAuthority16290.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16718.actual selector witness, LeftAuthority16290.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16722

namespace LeftBound16726
def owner : Owner := ⟨.program ⟨214⟩, ⟨18401⟩⟩
def transferEvent : Nat := 16726
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16724 .coefficient, .predecessor 1 16725 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16724 .coefficient)
      LeftBound16722.bound (LeftBound16722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16722.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16722.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16725 .coefficient)
      LeftAuthority16267.bound (LeftAuthority16267.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events063.exact16268RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16267.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16267.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16722.bound, LeftAuthority16267.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16722.bound, LeftAuthority16267.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16722.actual selector witness, LeftAuthority16267.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16726

namespace LeftBound16729
def owner : Owner := ⟨.program ⟨214⟩, ⟨18402⟩⟩
def transferEvent : Nat := 16729
def frameStart : Nat := 16225
def rule : BoundRule := .identity (.predecessor 0 16728 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16728 .coefficient)
      LeftBound16726.bound (LeftBound16726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16726.derived selector witness)

def rawBound : CoeffClass := LeftBound16726.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16726.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound16726.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound16729

namespace LeftBound16746
def owner : Owner := ⟨.program ⟨214⟩, ⟨18663⟩⟩
def transferEvent : Nat := 16746
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16744 .coefficient, .predecessor 1 16745 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16744 .coefficient)
      LeftBound16729.bound (LeftBound16729.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound16729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16745 .coefficient)
      LeftAuthority16742.bound (LeftAuthority16742.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority16742.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16729.bound, LeftAuthority16742.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16729.bound, LeftAuthority16742.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16729.actual selector witness, LeftAuthority16742.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16746

namespace LeftBound16749
def owner : Owner := ⟨.program ⟨214⟩, ⟨18664⟩⟩
def transferEvent : Nat := 16749
def frameStart : Nat := 16225
def rule : BoundRule := .identity (.predecessor 0 16748 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16748 .coefficient)
      LeftBound16746.bound (LeftBound16746.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound16746.derived selector witness)

def rawBound : CoeffClass := LeftBound16746.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16746.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound16746.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound16749

namespace LeftBound16755
def owner : Owner := ⟨.program ⟨214⟩, ⟨18665⟩⟩
def transferEvent : Nat := 16755
def frameStart : Nat := 16225
def rule : BoundRule := .product (.predecessor 0 16753 .coefficient) (.predecessor 1 16754 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16753 .coefficient)
      LeftAuthority16751.bound (LeftAuthority16751.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16752RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16751.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16751.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16754 .coefficient)
      LeftBound16749.bound (LeftBound16749.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16749.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16749.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority16751.bound LeftBound16749.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority16751.bound, LeftBound16749.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority16751.actual selector witness) * (LeftBound16749.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound16755

namespace LeftBound16831
def owner : Owner := ⟨.program ⟨214⟩, ⟨6795⟩⟩
def transferEvent : Nat := 16831
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16829 .coefficient, .predecessor 1 16830 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16829 .coefficient)
      LeftAuthority16827.bound (LeftAuthority16827.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16828RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16827.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16827.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16830 .coefficient)
      LeftAuthority16824.bound (LeftAuthority16824.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16825RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16824.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16824.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority16827.bound, LeftAuthority16824.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority16827.bound, LeftAuthority16824.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority16827.actual selector witness, LeftAuthority16824.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16831

namespace LeftBound16835
def owner : Owner := ⟨.program ⟨214⟩, ⟨6796⟩⟩
def transferEvent : Nat := 16835
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16833 .coefficient, .predecessor 1 16834 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16833 .coefficient)
      LeftBound16831.bound (LeftBound16831.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16832RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16831.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16831.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16834 .coefficient)
      LeftAuthority16821.bound (LeftAuthority16821.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16822RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16821.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16821.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16831.bound, LeftAuthority16821.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16831.bound, LeftAuthority16821.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16831.actual selector witness, LeftAuthority16821.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16835

namespace LeftBound16839
def owner : Owner := ⟨.program ⟨214⟩, ⟨6797⟩⟩
def transferEvent : Nat := 16839
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16837 .coefficient, .predecessor 1 16838 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16837 .coefficient)
      LeftBound16835.bound (LeftBound16835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16835.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16835.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16838 .coefficient)
      LeftAuthority16818.bound (LeftAuthority16818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16819RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16818.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16818.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16835.bound, LeftAuthority16818.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16835.bound, LeftAuthority16818.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16835.actual selector witness, LeftAuthority16818.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16839

namespace LeftBound16843
def owner : Owner := ⟨.program ⟨214⟩, ⟨6798⟩⟩
def transferEvent : Nat := 16843
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16841 .coefficient, .predecessor 1 16842 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16841 .coefficient)
      LeftBound16839.bound (LeftBound16839.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16839.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16839.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16842 .coefficient)
      LeftAuthority16815.bound (LeftAuthority16815.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16816RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16815.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16815.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16839.bound, LeftAuthority16815.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16839.bound, LeftAuthority16815.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16839.actual selector witness, LeftAuthority16815.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16843

namespace LeftBound16847
def owner : Owner := ⟨.program ⟨214⟩, ⟨6799⟩⟩
def transferEvent : Nat := 16847
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16845 .coefficient, .predecessor 1 16846 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16845 .coefficient)
      LeftBound16843.bound (LeftBound16843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16844RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16843.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16843.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16846 .coefficient)
      LeftAuthority16812.bound (LeftAuthority16812.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16813RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16812.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16812.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16843.bound, LeftAuthority16812.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16843.bound, LeftAuthority16812.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16843.actual selector witness, LeftAuthority16812.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16847

namespace LeftBound16851
def owner : Owner := ⟨.program ⟨214⟩, ⟨6800⟩⟩
def transferEvent : Nat := 16851
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16849 .coefficient, .predecessor 1 16850 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16849 .coefficient)
      LeftBound16847.bound (LeftBound16847.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16848RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16847.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16847.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16850 .coefficient)
      LeftAuthority16809.bound (LeftAuthority16809.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16810RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16809.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16809.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16847.bound, LeftAuthority16809.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16847.bound, LeftAuthority16809.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16847.actual selector witness, LeftAuthority16809.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16851

namespace LeftBound16855
def owner : Owner := ⟨.program ⟨214⟩, ⟨6801⟩⟩
def transferEvent : Nat := 16855
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16853 .coefficient, .predecessor 1 16854 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16853 .coefficient)
      LeftBound16851.bound (LeftBound16851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16851.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16851.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16854 .coefficient)
      LeftAuthority16806.bound (LeftAuthority16806.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16807RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16806.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16806.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16851.bound, LeftAuthority16806.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16851.bound, LeftAuthority16806.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16851.actual selector witness, LeftAuthority16806.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16855

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
