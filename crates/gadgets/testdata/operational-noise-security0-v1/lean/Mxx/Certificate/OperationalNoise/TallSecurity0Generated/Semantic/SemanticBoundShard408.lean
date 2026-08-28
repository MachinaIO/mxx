import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard407

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound60713
def owner : Owner := ⟨.program ⟨214⟩, ⟨6796⟩⟩
def transferEvent : Nat := 60713
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60711 .coefficient, .predecessor 1 60712 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60711 .coefficient)
      LeftBound60709.bound (LeftBound60709.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60710RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60709.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60709.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60712 .coefficient)
      LeftAuthority60699.bound (LeftAuthority60699.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60699.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60699.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60709.bound, LeftAuthority60699.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60709.bound, LeftAuthority60699.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60709.actual selector witness, LeftAuthority60699.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60713

namespace LeftBound60717
def owner : Owner := ⟨.program ⟨214⟩, ⟨6797⟩⟩
def transferEvent : Nat := 60717
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60715 .coefficient, .predecessor 1 60716 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60715 .coefficient)
      LeftBound60713.bound (LeftBound60713.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60714RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60713.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60713.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60716 .coefficient)
      LeftAuthority60696.bound (LeftAuthority60696.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60697RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60696.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60696.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60713.bound, LeftAuthority60696.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60713.bound, LeftAuthority60696.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60713.actual selector witness, LeftAuthority60696.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60717

namespace LeftBound60721
def owner : Owner := ⟨.program ⟨214⟩, ⟨6798⟩⟩
def transferEvent : Nat := 60721
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60719 .coefficient, .predecessor 1 60720 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60719 .coefficient)
      LeftBound60717.bound (LeftBound60717.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60718RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60717.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60717.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60720 .coefficient)
      LeftAuthority60693.bound (LeftAuthority60693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60694RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60693.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60693.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60717.bound, LeftAuthority60693.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60717.bound, LeftAuthority60693.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60717.actual selector witness, LeftAuthority60693.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60721

namespace LeftBound60725
def owner : Owner := ⟨.program ⟨214⟩, ⟨6799⟩⟩
def transferEvent : Nat := 60725
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60723 .coefficient, .predecessor 1 60724 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60723 .coefficient)
      LeftBound60721.bound (LeftBound60721.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60722RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60721.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60721.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60724 .coefficient)
      LeftAuthority60690.bound (LeftAuthority60690.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60691RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60690.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60690.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60721.bound, LeftAuthority60690.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60721.bound, LeftAuthority60690.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60721.actual selector witness, LeftAuthority60690.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60725

namespace LeftBound60729
def owner : Owner := ⟨.program ⟨214⟩, ⟨6800⟩⟩
def transferEvent : Nat := 60729
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60727 .coefficient, .predecessor 1 60728 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60727 .coefficient)
      LeftBound60725.bound (LeftBound60725.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60726RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60725.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60725.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60728 .coefficient)
      LeftAuthority60687.bound (LeftAuthority60687.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60688RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60687.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60687.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60725.bound, LeftAuthority60687.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60725.bound, LeftAuthority60687.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60725.actual selector witness, LeftAuthority60687.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60729

namespace LeftBound60733
def owner : Owner := ⟨.program ⟨214⟩, ⟨6801⟩⟩
def transferEvent : Nat := 60733
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60731 .coefficient, .predecessor 1 60732 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60731 .coefficient)
      LeftBound60729.bound (LeftBound60729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60730RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60729.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60732 .coefficient)
      LeftAuthority60684.bound (LeftAuthority60684.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60685RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60684.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60684.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60729.bound, LeftAuthority60684.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60729.bound, LeftAuthority60684.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60729.actual selector witness, LeftAuthority60684.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60733

namespace LeftBound60737
def owner : Owner := ⟨.program ⟨214⟩, ⟨6802⟩⟩
def transferEvent : Nat := 60737
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60735 .coefficient, .predecessor 1 60736 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60735 .coefficient)
      LeftBound60733.bound (LeftBound60733.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60734RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60733.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60733.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60736 .coefficient)
      LeftAuthority60681.bound (LeftAuthority60681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60682RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60681.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60681.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60733.bound, LeftAuthority60681.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60733.bound, LeftAuthority60681.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60733.actual selector witness, LeftAuthority60681.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60737

namespace LeftBound60741
def owner : Owner := ⟨.program ⟨214⟩, ⟨6803⟩⟩
def transferEvent : Nat := 60741
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60739 .coefficient, .predecessor 1 60740 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60739 .coefficient)
      LeftBound60737.bound (LeftBound60737.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60738RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60737.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60737.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60740 .coefficient)
      LeftAuthority60678.bound (LeftAuthority60678.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60679RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60678.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60678.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60737.bound, LeftAuthority60678.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60737.bound, LeftAuthority60678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60737.actual selector witness, LeftAuthority60678.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60741

namespace LeftBound60745
def owner : Owner := ⟨.program ⟨214⟩, ⟨6804⟩⟩
def transferEvent : Nat := 60745
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60743 .coefficient, .predecessor 1 60744 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60743 .coefficient)
      LeftBound60741.bound (LeftBound60741.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60742RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60741.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60741.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60744 .coefficient)
      LeftAuthority60675.bound (LeftAuthority60675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60676RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60675.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60675.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60741.bound, LeftAuthority60675.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60741.bound, LeftAuthority60675.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60741.actual selector witness, LeftAuthority60675.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60745

namespace LeftBound60749
def owner : Owner := ⟨.program ⟨214⟩, ⟨6805⟩⟩
def transferEvent : Nat := 60749
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60747 .coefficient, .predecessor 1 60748 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60747 .coefficient)
      LeftBound60745.bound (LeftBound60745.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60746RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60745.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60745.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60748 .coefficient)
      LeftAuthority60672.bound (LeftAuthority60672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60672.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60672.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60745.bound, LeftAuthority60672.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60745.bound, LeftAuthority60672.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60745.actual selector witness, LeftAuthority60672.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60749

namespace LeftBound60753
def owner : Owner := ⟨.program ⟨214⟩, ⟨6806⟩⟩
def transferEvent : Nat := 60753
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60751 .coefficient, .predecessor 1 60752 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60751 .coefficient)
      LeftBound60749.bound (LeftBound60749.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60749.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60749.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60752 .coefficient)
      LeftAuthority60669.bound (LeftAuthority60669.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60669.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60669.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60749.bound, LeftAuthority60669.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60749.bound, LeftAuthority60669.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60749.actual selector witness, LeftAuthority60669.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60753

namespace LeftBound60757
def owner : Owner := ⟨.program ⟨214⟩, ⟨6807⟩⟩
def transferEvent : Nat := 60757
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60755 .coefficient, .predecessor 1 60756 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60755 .coefficient)
      LeftBound60753.bound (LeftBound60753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60754RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60753.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60753.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60756 .coefficient)
      LeftAuthority60666.bound (LeftAuthority60666.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60667RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60666.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60666.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60753.bound, LeftAuthority60666.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60753.bound, LeftAuthority60666.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60753.actual selector witness, LeftAuthority60666.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60757

namespace LeftBound60761
def owner : Owner := ⟨.program ⟨214⟩, ⟨6808⟩⟩
def transferEvent : Nat := 60761
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60759 .coefficient, .predecessor 1 60760 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60759 .coefficient)
      LeftBound60757.bound (LeftBound60757.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60758RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60757.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60757.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60760 .coefficient)
      LeftAuthority60663.bound (LeftAuthority60663.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60664RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60663.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60663.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60757.bound, LeftAuthority60663.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60757.bound, LeftAuthority60663.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60757.actual selector witness, LeftAuthority60663.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60761

namespace LeftBound60765
def owner : Owner := ⟨.program ⟨214⟩, ⟨6809⟩⟩
def transferEvent : Nat := 60765
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60763 .coefficient, .predecessor 1 60764 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60763 .coefficient)
      LeftBound60761.bound (LeftBound60761.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60761.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60761.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60764 .coefficient)
      LeftAuthority60660.bound (LeftAuthority60660.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60660.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60660.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60761.bound, LeftAuthority60660.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60761.bound, LeftAuthority60660.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60761.actual selector witness, LeftAuthority60660.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60765

namespace LeftBound60769
def owner : Owner := ⟨.program ⟨214⟩, ⟨6810⟩⟩
def transferEvent : Nat := 60769
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60767 .coefficient, .predecessor 1 60768 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60767 .coefficient)
      LeftBound60765.bound (LeftBound60765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60766RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60765.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60765.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60768 .coefficient)
      LeftAuthority60657.bound (LeftAuthority60657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60658RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60657.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60657.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60765.bound, LeftAuthority60657.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60765.bound, LeftAuthority60657.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60765.actual selector witness, LeftAuthority60657.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60769

namespace LeftBound60773
def owner : Owner := ⟨.program ⟨214⟩, ⟨6811⟩⟩
def transferEvent : Nat := 60773
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60771 .coefficient, .predecessor 1 60772 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60771 .coefficient)
      LeftBound60769.bound (LeftBound60769.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60770RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60769.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60769.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60772 .coefficient)
      LeftAuthority60654.bound (LeftAuthority60654.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60654.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60654.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60769.bound, LeftAuthority60654.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60769.bound, LeftAuthority60654.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60769.actual selector witness, LeftAuthority60654.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60773

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
