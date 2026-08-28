import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard291
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard295
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard298
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard301

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound44651
def owner : Owner := ⟨.program ⟨214⟩, ⟨14840⟩⟩
def transferEvent : Nat := 44651
def frameStart : Nat := 44595
def rule : BoundRule := .sum [.predecessor 0 44649 .coefficient, .predecessor 1 44650 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44649 .coefficient)
      LeftBound44634.bound (LeftBound44634.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound44634.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44650 .coefficient)
      LeftAuthority44647.bound (LeftAuthority44647.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority44647.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44634.bound, LeftAuthority44647.bound]
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44634.bound, LeftAuthority44647.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44634.actual selector witness, LeftAuthority44647.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44651

namespace LeftBound44654
def owner : Owner := ⟨.program ⟨214⟩, ⟨14841⟩⟩
def transferEvent : Nat := 44654
def frameStart : Nat := 44595
def rule : BoundRule := .identity (.predecessor 0 44653 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44653 .coefficient)
      LeftBound44651.bound (LeftBound44651.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound44651.derived selector witness)

def rawBound : CoeffClass := LeftBound44651.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44651.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound44651.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound44654

namespace LeftBound44660
def owner : Owner := ⟨.program ⟨214⟩, ⟨14842⟩⟩
def transferEvent : Nat := 44660
def frameStart : Nat := 44595
def rule : BoundRule := .product (.predecessor 0 44658 .coefficient) (.predecessor 1 44659 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44658 .coefficient)
      LeftAuthority44656.bound (LeftAuthority44656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44657RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44656.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44656.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44659 .coefficient)
      LeftBound44654.bound (LeftBound44654.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44654.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44654.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority44656.bound LeftBound44654.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44656.bound, LeftBound44654.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority44656.actual selector witness) * (LeftBound44654.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44660

namespace LeftBound44668
def owner : Owner := ⟨.program ⟨214⟩, ⟨14843⟩⟩
def transferEvent : Nat := 44668
def frameStart : Nat := 44595
def rule : BoundRule := .sum [.predecessor 0 44666 .coefficient, .predecessor 1 44667 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44666 .coefficient)
      LeftAuthority44664.bound (LeftAuthority44664.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44664.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44664.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44667 .coefficient)
      LeftBound44660.bound (LeftBound44660.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44662RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44660.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44660.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority44664.bound, LeftBound44660.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44664.bound, LeftBound44660.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority44664.actual selector witness, LeftBound44660.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44668

namespace LeftBound44672
def owner : Owner := ⟨.program ⟨214⟩, ⟨26383⟩⟩
def transferEvent : Nat := 44672
def frameStart : Nat := 44595
def rule : BoundRule := .product (.predecessor 0 44670 .coefficient) (.predecessor 1 44671 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44670 .coefficient)
      LeftBound44668.bound (LeftBound44668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44669RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44668.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44671 .coefficient)
      LeftAuthority44645.bound (LeftAuthority44645.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44646RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44645.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44645.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound44668.bound LeftAuthority44645.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44668.bound, LeftAuthority44645.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound44668.actual selector witness) * (LeftAuthority44645.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44672

namespace LeftBound44683
def owner : Owner := ⟨.program ⟨214⟩, ⟨15272⟩⟩
def transferEvent : Nat := 44683
def frameStart : Nat := 44595
def rule : BoundRule := .product (.predecessor 0 44681 .coefficient) (.predecessor 1 44682 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44681 .coefficient)
      LeftAuthority44656.bound (LeftAuthority44656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44657RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44656.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44656.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44682 .coefficient)
      LeftAuthority44679.bound (LeftAuthority44679.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44680RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44679.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44679.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority44656.bound LeftAuthority44679.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44656.bound, LeftAuthority44679.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority44656.actual selector witness) * (LeftAuthority44679.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44683

namespace LeftBound44691
def owner : Owner := ⟨.program ⟨214⟩, ⟨15273⟩⟩
def transferEvent : Nat := 44691
def frameStart : Nat := 44595
def rule : BoundRule := .sum [.predecessor 0 44689 .coefficient, .predecessor 1 44690 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44689 .coefficient)
      LeftAuthority44687.bound (LeftAuthority44687.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44688RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44687.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44687.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44690 .coefficient)
      LeftBound44683.bound (LeftBound44683.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44685RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44683.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44683.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority44687.bound, LeftBound44683.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44687.bound, LeftBound44683.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority44687.actual selector witness, LeftBound44683.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44691

namespace LeftBound44695
def owner : Owner := ⟨.program ⟨214⟩, ⟨26386⟩⟩
def transferEvent : Nat := 44695
def frameStart : Nat := 44595
def rule : BoundRule := .sum [.predecessor 0 44693 .coefficient, .predecessor 1 44694 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44693 .coefficient)
      LeftBound44691.bound (LeftBound44691.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44692RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44691.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44691.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44694 .coefficient)
      LeftBound44672.bound (LeftBound44672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44672.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44672.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44691.bound, LeftBound44672.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44691.bound, LeftBound44672.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44691.actual selector witness, LeftBound44672.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44695

namespace LeftBound44708
def owner : Owner := ⟨.program ⟨214⟩, ⟨26385⟩⟩
def transferEvent : Nat := 44708
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44706 .coefficient, .predecessor 1 44707 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44706 .coefficient)
      LeftBound44537.bound (LeftBound44537.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44705RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44537.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44537.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44707 .coefficient)
      LeftBound44520.bound (LeftBound44520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44527RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44520.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44537.bound, LeftBound44520.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44537.bound, LeftBound44520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44537.actual selector witness, LeftBound44520.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44708

namespace LeftBound44711
def owner : Owner := ⟨.program ⟨214⟩, ⟨26385⟩⟩
def transferEvent : Nat := 44711
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44705 .summary, .result 44527 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44705 .summary)
      LeftBound44539.bound (LeftBound44539.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20403⟩⟩) (rawTerms := some (Proof.Events174.exact44705RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44527 .summary)
      LeftBound44522.bound (LeftBound44522.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26384⟩⟩) (rawTerms := some (Proof.Events173.exact44527RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44522.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44539.bound, LeftBound44522.bound]
def bound : CoeffClass := .finite ⟨1291889174379421642752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44539.bound, LeftBound44522.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44539.actual selector witness, LeftBound44522.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44711

namespace LeftBound44715
def owner : Owner := ⟨.program ⟨214⟩, ⟨26594⟩⟩
def transferEvent : Nat := 44715
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44713 .coefficient, .predecessor 1 44714 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44713 .coefficient)
      LeftBound44708.bound (LeftBound44708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44712RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44708.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44708.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44714 .coefficient)
      LeftBound44226.bound (LeftBound44226.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44230RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44226.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44226.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44708.bound, LeftBound44226.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44708.bound, LeftBound44226.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44708.actual selector witness, LeftBound44226.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44715

namespace LeftBound44716
def owner : Owner := ⟨.program ⟨214⟩, ⟨26594⟩⟩
def transferEvent : Nat := 44716
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44712 .summary, .result 44230 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44712 .summary)
      LeftBound44711.bound (LeftBound44711.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26385⟩⟩) (rawTerms := some (Proof.Events174.exact44712RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44711.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44230 .summary)
      LeftBound44229.bound (LeftBound44229.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26593⟩⟩) (rawTerms := some (Proof.Events172.exact44230RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44229.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44711.bound, LeftBound44229.bound]
def bound : CoeffClass := .finite ⟨2583789554981353578496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44711.bound, LeftBound44229.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44711.actual selector witness, LeftBound44229.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44716

namespace LeftBound44720
def owner : Owner := ⟨.program ⟨214⟩, ⟨26811⟩⟩
def transferEvent : Nat := 44720
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44718 .coefficient, .predecessor 1 44719 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44718 .coefficient)
      LeftBound44715.bound (LeftBound44715.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44715.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44715.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44719 .coefficient)
      LeftBound43744.bound (LeftBound43744.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43748RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43744.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43744.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44715.bound, LeftBound43744.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44715.bound, LeftBound43744.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44715.actual selector witness, LeftBound43744.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44720

namespace LeftBound44721
def owner : Owner := ⟨.program ⟨214⟩, ⟨26811⟩⟩
def transferEvent : Nat := 44721
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44717 .summary, .result 43748 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44717 .summary)
      LeftBound44716.bound (LeftBound44716.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26594⟩⟩) (rawTerms := some (Proof.Events174.exact44717RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44716.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43748 .summary)
      LeftBound43747.bound (LeftBound43747.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26810⟩⟩) (rawTerms := some (Proof.Events170.exact43748RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43747.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44716.bound, LeftBound43747.bound]
def bound : CoeffClass := .finite ⟨3875701141805795807232, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44716.bound, LeftBound43747.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44716.actual selector witness, LeftBound43747.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44721

namespace LeftBound44725
def owner : Owner := ⟨.program ⟨214⟩, ⟨27028⟩⟩
def transferEvent : Nat := 44725
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44723 .coefficient, .predecessor 1 44724 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44723 .coefficient)
      LeftBound44720.bound (LeftBound44720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44722RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44720.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44720.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44724 .coefficient)
      LeftBound43262.bound (LeftBound43262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43266RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43262.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43262.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44720.bound, LeftBound43262.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44720.bound, LeftBound43262.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44720.actual selector witness, LeftBound43262.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44725

namespace LeftBound44726
def owner : Owner := ⟨.program ⟨214⟩, ⟨27028⟩⟩
def transferEvent : Nat := 44726
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44722 .summary, .result 43266 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44722 .summary)
      LeftBound44721.bound (LeftBound44721.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26811⟩⟩) (rawTerms := some (Proof.Events174.exact44722RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44721.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43266 .summary)
      LeftBound43265.bound (LeftBound43265.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27027⟩⟩) (rawTerms := some (Proof.Events169.exact43266RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43265.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44721.bound, LeftBound43265.bound]
def bound : CoeffClass := .finite ⟨5167635141075258621952, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44721.bound, LeftBound43265.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44721.actual selector witness, LeftBound43265.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44726

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
