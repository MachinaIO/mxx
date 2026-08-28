import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard036
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard037
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard345

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound51624
def owner : Owner := ⟨.program ⟨214⟩, ⟨29835⟩⟩
def transferEvent : Nat := 51624
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 51618 .summary, .result 51440 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51618 .summary)
      LeftBound51452.bound (LeftBound51452.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22703⟩⟩) (rawTerms := some (Proof.Events201.exact51618RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51452.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51440 .summary)
      LeftBound51435.bound (LeftBound51435.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29834⟩⟩) (rawTerms := some (Proof.Events200.exact51440RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51435.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51452.bound, LeftBound51435.bound]
def bound : CoeffClass := .finite ⟨1292516722839998050304, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51452.bound, LeftBound51435.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51452.actual selector witness, LeftBound51435.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51624

namespace LeftBound51648
def owner : Owner := ⟨.program ⟨214⟩, ⟨12969⟩⟩
def transferEvent : Nat := 51648
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 51646 .coefficient) (.predecessor 1 51647 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51646 .coefficient)
      LeftAuthority2383.bound (LeftAuthority2383.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2384RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2383.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2383.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51647 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2383.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2383.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2383.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound51648

namespace LeftBound51653
def owner : Owner := ⟨.program ⟨214⟩, ⟨7282⟩⟩
def transferEvent : Nat := 51653
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 51651 .coefficient) (.predecessor 1 51652 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51651 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51652 .coefficient)
      LeftBound7473.bound (LeftBound7473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7473.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound7473.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound7473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound7473.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51653

namespace LeftBound51658
def owner : Owner := ⟨.program ⟨214⟩, ⟨12970⟩⟩
def transferEvent : Nat := 51658
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 51656 .coefficient, .predecessor 1 51657 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51656 .coefficient)
      LeftBound51653.bound (LeftBound51653.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51653.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51653.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51657 .coefficient)
      LeftBound51648.bound (LeftBound51648.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51650RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51648.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51648.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51653.bound, LeftBound51648.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51653.bound, LeftBound51648.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51653.actual selector witness, LeftBound51648.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51658

namespace LeftBound51662
def owner : Owner := ⟨.program ⟨214⟩, ⟨12971⟩⟩
def transferEvent : Nat := 51662
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 51660 .coefficient, .predecessor 1 51661 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51660 .coefficient)
      LeftBound51658.bound (LeftBound51658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51658.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51661 .coefficient)
      LeftBound7465.bound (LeftBound7465.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7466RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7465.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51658.bound, LeftBound7465.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51658.bound, LeftBound7465.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51658.actual selector witness, LeftBound7465.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51662

namespace LeftBound51663
def owner : Owner := ⟨.program ⟨214⟩, ⟨12971⟩⟩
def transferEvent : Nat := 51663
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨102⟩⟩]⟩ [⟨.result 7466 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7466 .coefficient)
      LeftBound7465.bound (LeftBound7465.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨102⟩⟩) (rawTerms := some (Proof.Events029.exact7466RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7465.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound7465.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7465.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound7465.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound51663

namespace LeftBound51668
def owner : Owner := ⟨.program ⟨214⟩, ⟨12972⟩⟩
def transferEvent : Nat := 51668
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 51666 .coefficient) (.predecessor 1 51667 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51666 .coefficient)
      LeftBound51662.bound (LeftBound51662.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51662.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51662.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51667 .coefficient)
      LeftAuthority2386.bound (LeftAuthority2386.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2386.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2386.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound51662.bound LeftAuthority2386.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51662.bound, LeftAuthority2386.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound51662.actual selector witness) * (LeftAuthority2386.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51668

namespace LeftBound51669
def owner : Owner := ⟨.program ⟨214⟩, ⟨12972⟩⟩
def transferEvent : Nat := 51669
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨10140⟩⟩], []⟩ [⟨.result 2387 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2387 .coefficient)
      LeftAuthority2386.bound (LeftAuthority2386.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨10140⟩⟩) (rawTerms := some (Proof.Events009.exact2387RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2386.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2386.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority2386.bound []
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2386.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority2386.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound51669

namespace LeftBound51670
def owner : Owner := ⟨.program ⟨214⟩, ⟨12972⟩⟩
def transferEvent : Nat := 51670
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 51665 .summary) (.transfer 51669) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51665 .summary)
      LeftBound51663.bound (LeftBound51663.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12971⟩⟩) (rawTerms := some (Proof.Events201.exact51665RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51663.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 51669)
      LeftBound51669.bound (LeftBound51669.actual selector witness) := by
  exact .transfer (LeftBound51669.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound51663.bound LeftBound51669.bound
def bound : CoeffClass := .finite ⟨43264, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51663.bound, LeftBound51669.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound51663.actual selector witness) * (LeftBound51669.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51670

namespace LeftBound51676
def owner : Owner := ⟨.program ⟨214⟩, ⟨10141⟩⟩
def transferEvent : Nat := 51676
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 51674 .coefficient) (.predecessor 1 51675 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51674 .coefficient)
      LeftAuthority2386.bound (LeftAuthority2386.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2386.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2386.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51675 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2386.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2386.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2386.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound51676

namespace LeftBound51681
def owner : Owner := ⟨.program ⟨214⟩, ⟨7262⟩⟩
def transferEvent : Nat := 51681
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 51679 .coefficient) (.predecessor 1 51680 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51679 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51680 .coefficient)
      LeftBound7514.bound (LeftBound7514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7514.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound7514.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound7514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound7514.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51681

namespace LeftBound51686
def owner : Owner := ⟨.program ⟨214⟩, ⟨10142⟩⟩
def transferEvent : Nat := 51686
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 51684 .coefficient, .predecessor 1 51685 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51684 .coefficient)
      LeftBound51681.bound (LeftBound51681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51681.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51681.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51685 .coefficient)
      LeftBound51676.bound (LeftBound51676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51678RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51676.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51676.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51681.bound, LeftBound51676.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51681.bound, LeftBound51676.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51681.actual selector witness, LeftBound51676.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51686

namespace LeftBound51690
def owner : Owner := ⟨.program ⟨214⟩, ⟨10143⟩⟩
def transferEvent : Nat := 51690
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 51688 .coefficient, .predecessor 1 51689 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51688 .coefficient)
      LeftBound51686.bound (LeftBound51686.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51687RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51686.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51686.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51689 .coefficient)
      LeftBound7506.bound (LeftBound7506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7507RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7506.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51686.bound, LeftBound7506.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51686.bound, LeftBound7506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51686.actual selector witness, LeftBound7506.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51690

namespace LeftBound51691
def owner : Owner := ⟨.program ⟨214⟩, ⟨10143⟩⟩
def transferEvent : Nat := 51691
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨82⟩⟩]⟩ [⟨.result 7507 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7507 .coefficient)
      LeftBound7506.bound (LeftBound7506.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨82⟩⟩) (rawTerms := some (Proof.Events029.exact7507RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7506.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound7506.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound7506.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound51691

namespace LeftBound51696
def owner : Owner := ⟨.program ⟨214⟩, ⟨10144⟩⟩
def transferEvent : Nat := 51696
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 51694 .coefficient) (.predecessor 1 51695 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51694 .coefficient)
      LeftBound51690.bound (LeftBound51690.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51690.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51690.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51695 .coefficient)
      LeftBound7503.bound (LeftBound7503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7503.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51690.bound LeftBound7503.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51690.bound, LeftBound7503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51690.actual selector witness) * (LeftBound7503.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51696

namespace LeftBound51697
def owner : Owner := ⟨.program ⟨214⟩, ⟨10144⟩⟩
def transferEvent : Nat := 51697
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩ [⟨.result 7500 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7500 .coefficient)
      LeftAuthority7499.bound (LeftAuthority7499.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7876⟩⟩) (rawTerms := some (Proof.Events029.exact7500RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7499.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7499.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority7499.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7499.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority7499.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound51697

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
