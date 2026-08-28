import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard516
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard517
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard518
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard520
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard521
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard522
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard524
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard525
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard535

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound79473
def owner : Owner := ⟨.program ⟨214⟩, ⟨27851⟩⟩
def transferEvent : Nat := 79473
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79471 .coefficient, .predecessor 1 79472 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79471 .coefficient)
      LeftBound79468.bound (LeftBound79468.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79470RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79468.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79468.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79472 .coefficient)
      LeftBound77917.bound (LeftBound77917.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events304.exact77924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77917.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77917.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79468.bound, LeftBound77917.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79468.bound, LeftBound77917.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79468.actual selector witness, LeftBound77917.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79473

namespace LeftBound79474
def owner : Owner := ⟨.program ⟨214⟩, ⟨27851⟩⟩
def transferEvent : Nat := 79474
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79470 .summary, .result 77924 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79470 .summary)
      LeftBound79469.bound (LeftBound79469.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27634⟩⟩) (rawTerms := some (Proof.Events310.exact79470RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79469.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77924 .summary)
      LeftBound77919.bound (LeftBound77919.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27850⟩⟩) (rawTerms := some (Proof.Events304.exact77924RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77919.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79469.bound, LeftBound77919.bound]
def bound : CoeffClass := .finite ⟨37932293507469318446662025268, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79469.bound, LeftBound77919.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79469.actual selector witness, LeftBound77919.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79474

namespace LeftBound79478
def owner : Owner := ⟨.program ⟨214⟩, ⟨28068⟩⟩
def transferEvent : Nat := 79478
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79476 .coefficient, .predecessor 1 79477 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79476 .coefficient)
      LeftBound79473.bound (LeftBound79473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79473.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79477 .coefficient)
      LeftBound77705.bound (LeftBound77705.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77712RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77705.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77705.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79473.bound, LeftBound77705.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79473.bound, LeftBound77705.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79473.actual selector witness, LeftBound77705.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79478

namespace LeftBound79479
def owner : Owner := ⟨.program ⟨214⟩, ⟨28068⟩⟩
def transferEvent : Nat := 79479
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79475 .summary, .result 77712 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79475 .summary)
      LeftBound79474.bound (LeftBound79474.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27851⟩⟩) (rawTerms := some (Proof.Events310.exact79475RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79474.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77712 .summary)
      LeftBound77707.bound (LeftBound77707.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28067⟩⟩) (rawTerms := some (Proof.Events303.exact77712RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77707.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79474.bound, LeftBound77707.bound]
def bound : CoeffClass := .finite ⟨42674369987986832655214706740, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79474.bound, LeftBound77707.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79474.actual selector witness, LeftBound77707.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79479

namespace LeftBound79483
def owner : Owner := ⟨.program ⟨214⟩, ⟨28285⟩⟩
def transferEvent : Nat := 79483
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79481 .coefficient, .predecessor 1 79482 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79481 .coefficient)
      LeftBound79478.bound (LeftBound79478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79478.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79478.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79482 .coefficient)
      LeftBound77493.bound (LeftBound77493.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77493.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77493.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79478.bound, LeftBound77493.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79478.bound, LeftBound77493.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79478.actual selector witness, LeftBound77493.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79483

namespace LeftBound79484
def owner : Owner := ⟨.program ⟨214⟩, ⟨28285⟩⟩
def transferEvent : Nat := 79484
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79480 .summary, .result 77500 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79480 .summary)
      LeftBound79479.bound (LeftBound79479.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28068⟩⟩) (rawTerms := some (Proof.Events310.exact79480RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79479.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77500 .summary)
      LeftBound77495.bound (LeftBound77495.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28284⟩⟩) (rawTerms := some (Proof.Events302.exact77500RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77495.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79479.bound, LeftBound77495.bound]
def bound : CoeffClass := .finite ⟨47416693230599820876439355444, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79479.bound, LeftBound77495.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79479.actual selector witness, LeftBound77495.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79484

namespace LeftBound79488
def owner : Owner := ⟨.program ⟨214⟩, ⟨28502⟩⟩
def transferEvent : Nat := 79488
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79486 .coefficient, .predecessor 1 79487 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79486 .coefficient)
      LeftBound79483.bound (LeftBound79483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79485RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79483.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79483.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79487 .coefficient)
      LeftBound77281.bound (LeftBound77281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events301.exact77288RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77281.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77281.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79483.bound, LeftBound77281.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79483.bound, LeftBound77281.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79483.actual selector witness, LeftBound77281.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79488

namespace LeftBound79489
def owner : Owner := ⟨.program ⟨214⟩, ⟨28502⟩⟩
def transferEvent : Nat := 79489
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79485 .summary, .result 77288 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79485 .summary)
      LeftBound79484.bound (LeftBound79484.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28285⟩⟩) (rawTerms := some (Proof.Events310.exact79485RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79484.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77288 .summary)
      LeftBound77283.bound (LeftBound77283.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28501⟩⟩) (rawTerms := some (Proof.Events301.exact77288RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77283.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79484.bound, LeftBound77283.bound]
def bound : CoeffClass := .finite ⟨52159098727244633768554659892, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79484.bound, LeftBound77283.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79484.actual selector witness, LeftBound77283.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79489

namespace LeftBound79493
def owner : Owner := ⟨.program ⟨214⟩, ⟨28719⟩⟩
def transferEvent : Nat := 79493
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79491 .coefficient, .predecessor 1 79492 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79491 .coefficient)
      LeftBound79488.bound (LeftBound79488.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79488.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79488.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79492 .coefficient)
      LeftBound77069.bound (LeftBound77069.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events301.exact77076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77069.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77069.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79488.bound, LeftBound77069.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79488.bound, LeftBound77069.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79488.actual selector witness, LeftBound77069.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79493

namespace LeftBound79494
def owner : Owner := ⟨.program ⟨214⟩, ⟨28719⟩⟩
def transferEvent : Nat := 79494
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79490 .summary, .result 77076 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79490 .summary)
      LeftBound79489.bound (LeftBound79489.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28502⟩⟩) (rawTerms := some (Proof.Events310.exact79490RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79489.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77076 .summary)
      LeftBound77071.bound (LeftBound77071.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28718⟩⟩) (rawTerms := some (Proof.Events301.exact77076RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77071.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79489.bound, LeftBound77071.bound]
def bound : CoeffClass := .finite ⟨56901750985984920673341931572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79489.bound, LeftBound77071.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79489.actual selector witness, LeftBound77071.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79494

namespace LeftBound79498
def owner : Owner := ⟨.program ⟨214⟩, ⟨28936⟩⟩
def transferEvent : Nat := 79498
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79496 .coefficient, .predecessor 1 79497 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79496 .coefficient)
      LeftBound79493.bound (LeftBound79493.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79493.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79497 .coefficient)
      LeftBound76857.bound (LeftBound76857.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact76864RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76857.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76857.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79493.bound, LeftBound76857.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79493.bound, LeftBound76857.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79493.actual selector witness, LeftBound76857.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79498

namespace LeftBound79499
def owner : Owner := ⟨.program ⟨214⟩, ⟨28936⟩⟩
def transferEvent : Nat := 79499
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79495 .summary, .result 76864 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79495 .summary)
      LeftBound79494.bound (LeftBound79494.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28719⟩⟩) (rawTerms := some (Proof.Events310.exact79495RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79494.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76864 .summary)
      LeftBound76859.bound (LeftBound76859.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28935⟩⟩) (rawTerms := some (Proof.Events300.exact76864RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76859.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79494.bound, LeftBound76859.bound]
def bound : CoeffClass := .finite ⟨61644567752788856919910514740, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79494.bound, LeftBound76859.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79494.actual selector witness, LeftBound76859.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79499

namespace LeftBound79503
def owner : Owner := ⟨.program ⟨214⟩, ⟨29153⟩⟩
def transferEvent : Nat := 79503
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79501 .coefficient, .predecessor 1 79502 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79501 .coefficient)
      LeftBound79498.bound (LeftBound79498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79498.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79498.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79502 .coefficient)
      LeftBound76645.bound (LeftBound76645.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76652RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76645.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76645.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79498.bound, LeftBound76645.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79498.bound, LeftBound76645.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79498.actual selector witness, LeftBound76645.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79503

namespace LeftBound79504
def owner : Owner := ⟨.program ⟨214⟩, ⟨29153⟩⟩
def transferEvent : Nat := 79504
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79500 .summary, .result 76652 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79500 .summary)
      LeftBound79499.bound (LeftBound79499.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28936⟩⟩) (rawTerms := some (Proof.Events310.exact79500RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79499.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76652 .summary)
      LeftBound76647.bound (LeftBound76647.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29152⟩⟩) (rawTerms := some (Proof.Events299.exact76652RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76647.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79499.bound, LeftBound76647.bound]
def bound : CoeffClass := .finite ⟨66387466773624617837369753652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79499.bound, LeftBound76647.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79499.actual selector witness, LeftBound76647.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79504

namespace LeftBound79508
def owner : Owner := ⟨.program ⟨214⟩, ⟨29370⟩⟩
def transferEvent : Nat := 79508
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79506 .coefficient, .predecessor 1 79507 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79506 .coefficient)
      LeftBound79503.bound (LeftBound79503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79505RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79503.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79507 .coefficient)
      LeftBound76433.bound (LeftBound76433.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76440RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76433.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76433.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79503.bound, LeftBound76433.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79503.bound, LeftBound76433.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79503.actual selector witness, LeftBound76433.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79508

namespace LeftBound79509
def owner : Owner := ⟨.program ⟨214⟩, ⟨29370⟩⟩
def transferEvent : Nat := 79509
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79505 .summary, .result 76440 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79505 .summary)
      LeftBound79504.bound (LeftBound79504.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29153⟩⟩) (rawTerms := some (Proof.Events310.exact79505RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79504.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76440 .summary)
      LeftBound76435.bound (LeftBound76435.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29369⟩⟩) (rawTerms := some (Proof.Events298.exact76440RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76435.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79504.bound, LeftBound76435.bound]
def bound : CoeffClass := .finite ⟨71130530302524028096610304052, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79504.bound, LeftBound76435.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79504.actual selector witness, LeftBound76435.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79509

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
