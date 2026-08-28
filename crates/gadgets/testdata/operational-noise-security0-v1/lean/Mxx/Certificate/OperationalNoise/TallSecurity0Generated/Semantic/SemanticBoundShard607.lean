import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard574
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard577
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard581
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard585
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard588
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard592
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard595
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard596
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard599
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard603
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard606

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound88555
def owner : Owner := ⟨.program ⟨214⟩, ⟨26568⟩⟩
def transferEvent : Nat := 88555
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88551 .summary, .result 88071 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88551 .summary)
      LeftBound88550.bound (LeftBound88550.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26361⟩⟩) (rawTerms := some (Proof.Events345.exact88551RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88550.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88071 .summary)
      LeftBound88070.bound (LeftBound88070.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26567⟩⟩) (rawTerms := some (Proof.Events344.exact88071RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88070.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88550.bound, LeftBound88070.bound]
def bound : CoeffClass := .finite ⟨2583789554981353578496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88550.bound, LeftBound88070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88550.actual selector witness, LeftBound88070.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88555

namespace LeftBound88559
def owner : Owner := ⟨.program ⟨214⟩, ⟨26785⟩⟩
def transferEvent : Nat := 88559
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88557 .coefficient, .predecessor 1 88558 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88557 .coefficient)
      LeftBound88554.bound (LeftBound88554.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88556RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88554.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88554.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88558 .coefficient)
      LeftBound87587.bound (LeftBound87587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87591RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87587.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87587.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88554.bound, LeftBound87587.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88554.bound, LeftBound87587.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88554.actual selector witness, LeftBound87587.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88559

namespace LeftBound88560
def owner : Owner := ⟨.program ⟨214⟩, ⟨26785⟩⟩
def transferEvent : Nat := 88560
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88556 .summary, .result 87591 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88556 .summary)
      LeftBound88555.bound (LeftBound88555.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26568⟩⟩) (rawTerms := some (Proof.Events345.exact88556RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88555.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87591 .summary)
      LeftBound87590.bound (LeftBound87590.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26784⟩⟩) (rawTerms := some (Proof.Events342.exact87591RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87590.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88555.bound, LeftBound87590.bound]
def bound : CoeffClass := .finite ⟨3875701141805795807232, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88555.bound, LeftBound87590.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88555.actual selector witness, LeftBound87590.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88560

namespace LeftBound88564
def owner : Owner := ⟨.program ⟨214⟩, ⟨27002⟩⟩
def transferEvent : Nat := 88564
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88562 .coefficient, .predecessor 1 88563 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88562 .coefficient)
      LeftBound88559.bound (LeftBound88559.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88559.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88563 .coefficient)
      LeftBound87107.bound (LeftBound87107.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87111RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87107.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87107.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88559.bound, LeftBound87107.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88559.bound, LeftBound87107.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88559.actual selector witness, LeftBound87107.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88564

namespace LeftBound88565
def owner : Owner := ⟨.program ⟨214⟩, ⟨27002⟩⟩
def transferEvent : Nat := 88565
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88561 .summary, .result 87111 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88561 .summary)
      LeftBound88560.bound (LeftBound88560.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26785⟩⟩) (rawTerms := some (Proof.Events345.exact88561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88560.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87111 .summary)
      LeftBound87110.bound (LeftBound87110.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27001⟩⟩) (rawTerms := some (Proof.Events340.exact87111RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87110.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88560.bound, LeftBound87110.bound]
def bound : CoeffClass := .finite ⟨5167635141075258621952, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88560.bound, LeftBound87110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88560.actual selector witness, LeftBound87110.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88565

namespace LeftBound88569
def owner : Owner := ⟨.program ⟨214⟩, ⟨27219⟩⟩
def transferEvent : Nat := 88569
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88567 .coefficient, .predecessor 1 88568 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88567 .coefficient)
      LeftBound88564.bound (LeftBound88564.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88566RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88564.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88564.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88568 .coefficient)
      LeftBound86627.bound (LeftBound86627.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events338.exact86631RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86627.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86627.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88564.bound, LeftBound86627.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88564.bound, LeftBound86627.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88564.actual selector witness, LeftBound86627.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88569

namespace LeftBound88570
def owner : Owner := ⟨.program ⟨214⟩, ⟨27219⟩⟩
def transferEvent : Nat := 88570
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88566 .summary, .result 86631 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88566 .summary)
      LeftBound88565.bound (LeftBound88565.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27002⟩⟩) (rawTerms := some (Proof.Events345.exact88566RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88565.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86631 .summary)
      LeftBound86630.bound (LeftBound86630.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27218⟩⟩) (rawTerms := some (Proof.Events338.exact86631RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound86630.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88565.bound, LeftBound86630.bound]
def bound : CoeffClass := .finite ⟨6459613965234762608640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88565.bound, LeftBound86630.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88565.actual selector witness, LeftBound86630.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88570

namespace LeftBound88574
def owner : Owner := ⟨.program ⟨214⟩, ⟨27436⟩⟩
def transferEvent : Nat := 88574
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88572 .coefficient, .predecessor 1 88573 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88572 .coefficient)
      LeftBound88569.bound (LeftBound88569.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88571RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88569.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88569.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88573 .coefficient)
      LeftBound86147.bound (LeftBound86147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86151RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86147.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86147.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88569.bound, LeftBound86147.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88569.bound, LeftBound86147.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88569.actual selector witness, LeftBound86147.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88574

namespace LeftBound88575
def owner : Owner := ⟨.program ⟨214⟩, ⟨27436⟩⟩
def transferEvent : Nat := 88575
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88571 .summary, .result 86151 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88571 .summary)
      LeftBound88570.bound (LeftBound88570.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27219⟩⟩) (rawTerms := some (Proof.Events345.exact88571RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88570.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86151 .summary)
      LeftBound86150.bound (LeftBound86150.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27435⟩⟩) (rawTerms := some (Proof.Events336.exact86151RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound86150.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88570.bound, LeftBound86150.bound]
def bound : CoeffClass := .finite ⟨7751615201839287181312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88570.bound, LeftBound86150.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88570.actual selector witness, LeftBound86150.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88575

namespace LeftBound88579
def owner : Owner := ⟨.program ⟨214⟩, ⟨27653⟩⟩
def transferEvent : Nat := 88579
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88577 .coefficient, .predecessor 1 88578 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88577 .coefficient)
      LeftBound88574.bound (LeftBound88574.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events346.exact88576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88574.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88574.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88578 .coefficient)
      LeftBound85667.bound (LeftBound85667.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85671RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85667.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85667.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88574.bound, LeftBound85667.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88574.bound, LeftBound85667.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88574.actual selector witness, LeftBound85667.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88579

namespace LeftBound88580
def owner : Owner := ⟨.program ⟨214⟩, ⟨27653⟩⟩
def transferEvent : Nat := 88580
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88576 .summary, .result 85671 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88576 .summary)
      LeftBound88575.bound (LeftBound88575.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27436⟩⟩) (rawTerms := some (Proof.Events346.exact88576RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88575.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85671 .summary)
      LeftBound85670.bound (LeftBound85670.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27652⟩⟩) (rawTerms := some (Proof.Events334.exact85671RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound85670.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88575.bound, LeftBound85670.bound]
def bound : CoeffClass := .finite ⟨9043661263333852925952, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88575.bound, LeftBound85670.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88575.actual selector witness, LeftBound85670.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88580

namespace LeftBound88584
def owner : Owner := ⟨.program ⟨214⟩, ⟨27870⟩⟩
def transferEvent : Nat := 88584
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88582 .coefficient, .predecessor 1 88583 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88582 .coefficient)
      LeftBound88579.bound (LeftBound88579.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events346.exact88581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88579.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88579.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88583 .coefficient)
      LeftBound85187.bound (LeftBound85187.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85187.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85187.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88579.bound, LeftBound85187.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88579.bound, LeftBound85187.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88579.actual selector witness, LeftBound85187.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88584

namespace LeftBound88585
def owner : Owner := ⟨.program ⟨214⟩, ⟨27870⟩⟩
def transferEvent : Nat := 88585
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88581 .summary, .result 85191 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88581 .summary)
      LeftBound88580.bound (LeftBound88580.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27653⟩⟩) (rawTerms := some (Proof.Events346.exact88581RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88580.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85191 .summary)
      LeftBound85190.bound (LeftBound85190.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27869⟩⟩) (rawTerms := some (Proof.Events332.exact85191RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound85190.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88580.bound, LeftBound85190.bound]
def bound : CoeffClass := .finite ⟨10335729737273439256576, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88580.bound, LeftBound85190.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88580.actual selector witness, LeftBound85190.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88585

namespace LeftBound88589
def owner : Owner := ⟨.program ⟨214⟩, ⟨28087⟩⟩
def transferEvent : Nat := 88589
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88587 .coefficient, .predecessor 1 88588 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88587 .coefficient)
      LeftBound88584.bound (LeftBound88584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events346.exact88586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88584.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88588 .coefficient)
      LeftBound84707.bound (LeftBound84707.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84711RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84707.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84707.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88584.bound, LeftBound84707.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88584.bound, LeftBound84707.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88584.actual selector witness, LeftBound84707.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88589

namespace LeftBound88590
def owner : Owner := ⟨.program ⟨214⟩, ⟨28087⟩⟩
def transferEvent : Nat := 88590
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88586 .summary, .result 84711 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88586 .summary)
      LeftBound88585.bound (LeftBound88585.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27870⟩⟩) (rawTerms := some (Proof.Events346.exact88586RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88585.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84711 .summary)
      LeftBound84710.bound (LeftBound84710.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28086⟩⟩) (rawTerms := some (Proof.Events330.exact84711RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84710.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88585.bound, LeftBound84710.bound]
def bound : CoeffClass := .finite ⟨11627843036103066759168, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88585.bound, LeftBound84710.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88585.actual selector witness, LeftBound84710.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88590

namespace LeftBound88594
def owner : Owner := ⟨.program ⟨214⟩, ⟨28304⟩⟩
def transferEvent : Nat := 88594
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88592 .coefficient, .predecessor 1 88593 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88592 .coefficient)
      LeftBound88589.bound (LeftBound88589.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events346.exact88591RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88589.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88589.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88593 .coefficient)
      LeftBound84227.bound (LeftBound84227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84231RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84227.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84227.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88589.bound, LeftBound84227.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88589.bound, LeftBound84227.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88589.actual selector witness, LeftBound84227.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88594

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
