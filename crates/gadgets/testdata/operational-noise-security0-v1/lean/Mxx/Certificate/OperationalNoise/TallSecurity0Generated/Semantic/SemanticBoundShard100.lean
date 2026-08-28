import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard060
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard064
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard068
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard072
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard075
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard076
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard079
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard083
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard087
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard099

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound15473
def owner : Owner := ⟨.program ⟨214⟩, ⟨27054⟩⟩
def transferEvent : Nat := 15473
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 15469 .summary, .result 13956 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15469 .summary)
      LeftBound15468.bound (LeftBound15468.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26837⟩⟩) (rawTerms := some (Proof.Events060.exact15469RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15468.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13956 .summary)
      LeftBound13955.bound (LeftBound13955.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27053⟩⟩) (rawTerms := some (Proof.Events054.exact13956RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13955.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15468.bound, LeftBound13955.bound]
def bound : CoeffClass := .finite ⟨5167635141075258621952, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15468.bound, LeftBound13955.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15468.actual selector witness, LeftBound13955.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15473

namespace LeftBound15477
def owner : Owner := ⟨.program ⟨214⟩, ⟨27271⟩⟩
def transferEvent : Nat := 15477
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15475 .coefficient, .predecessor 1 15476 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15475 .coefficient)
      LeftBound15472.bound (LeftBound15472.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15472.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15472.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15476 .coefficient)
      LeftBound13451.bound (LeftBound13451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13455RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13451.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13451.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15472.bound, LeftBound13451.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15472.bound, LeftBound13451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15472.actual selector witness, LeftBound13451.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15477

namespace LeftBound15478
def owner : Owner := ⟨.program ⟨214⟩, ⟨27271⟩⟩
def transferEvent : Nat := 15478
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 15474 .summary, .result 13455 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15474 .summary)
      LeftBound15473.bound (LeftBound15473.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27054⟩⟩) (rawTerms := some (Proof.Events060.exact15474RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15473.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13455 .summary)
      LeftBound13454.bound (LeftBound13454.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27270⟩⟩) (rawTerms := some (Proof.Events052.exact13455RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13454.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15473.bound, LeftBound13454.bound]
def bound : CoeffClass := .finite ⟨6459613965234762608640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15473.bound, LeftBound13454.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15473.actual selector witness, LeftBound13454.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15478

namespace LeftBound15482
def owner : Owner := ⟨.program ⟨214⟩, ⟨27488⟩⟩
def transferEvent : Nat := 15482
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15480 .coefficient, .predecessor 1 15481 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15480 .coefficient)
      LeftBound15477.bound (LeftBound15477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15479RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15477.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15481 .coefficient)
      LeftBound12950.bound (LeftBound12950.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12954RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12950.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12950.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15477.bound, LeftBound12950.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15477.bound, LeftBound12950.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15477.actual selector witness, LeftBound12950.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15482

namespace LeftBound15483
def owner : Owner := ⟨.program ⟨214⟩, ⟨27488⟩⟩
def transferEvent : Nat := 15483
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 15479 .summary, .result 12954 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15479 .summary)
      LeftBound15478.bound (LeftBound15478.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27271⟩⟩) (rawTerms := some (Proof.Events060.exact15479RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15478.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12954 .summary)
      LeftBound12953.bound (LeftBound12953.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27487⟩⟩) (rawTerms := some (Proof.Events050.exact12954RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound12953.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15478.bound, LeftBound12953.bound]
def bound : CoeffClass := .finite ⟨7751615201839287181312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15478.bound, LeftBound12953.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15478.actual selector witness, LeftBound12953.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15483

namespace LeftBound15487
def owner : Owner := ⟨.program ⟨214⟩, ⟨27705⟩⟩
def transferEvent : Nat := 15487
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15485 .coefficient, .predecessor 1 15486 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15485 .coefficient)
      LeftBound15482.bound (LeftBound15482.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15482.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15482.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15486 .coefficient)
      LeftBound12449.bound (LeftBound12449.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12453RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12449.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12449.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15482.bound, LeftBound12449.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15482.bound, LeftBound12449.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15482.actual selector witness, LeftBound12449.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15487

namespace LeftBound15488
def owner : Owner := ⟨.program ⟨214⟩, ⟨27705⟩⟩
def transferEvent : Nat := 15488
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 15484 .summary, .result 12453 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15484 .summary)
      LeftBound15483.bound (LeftBound15483.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27488⟩⟩) (rawTerms := some (Proof.Events060.exact15484RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15483.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12453 .summary)
      LeftBound12452.bound (LeftBound12452.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27704⟩⟩) (rawTerms := some (Proof.Events048.exact12453RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound12452.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15483.bound, LeftBound12452.bound]
def bound : CoeffClass := .finite ⟨9043661263333852925952, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15483.bound, LeftBound12452.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15483.actual selector witness, LeftBound12452.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15488

namespace LeftBound15492
def owner : Owner := ⟨.program ⟨214⟩, ⟨27922⟩⟩
def transferEvent : Nat := 15492
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15490 .coefficient, .predecessor 1 15491 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15490 .coefficient)
      LeftBound15487.bound (LeftBound15487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15489RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15487.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15487.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15491 .coefficient)
      LeftBound11948.bound (LeftBound11948.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11952RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11948.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11948.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15487.bound, LeftBound11948.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15487.bound, LeftBound11948.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15487.actual selector witness, LeftBound11948.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15492

namespace LeftBound15493
def owner : Owner := ⟨.program ⟨214⟩, ⟨27922⟩⟩
def transferEvent : Nat := 15493
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 15489 .summary, .result 11952 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15489 .summary)
      LeftBound15488.bound (LeftBound15488.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27705⟩⟩) (rawTerms := some (Proof.Events060.exact15489RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15488.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11952 .summary)
      LeftBound11951.bound (LeftBound11951.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27921⟩⟩) (rawTerms := some (Proof.Events046.exact11952RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound11951.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15488.bound, LeftBound11951.bound]
def bound : CoeffClass := .finite ⟨10335729737273439256576, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15488.bound, LeftBound11951.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15488.actual selector witness, LeftBound11951.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15493

namespace LeftBound15497
def owner : Owner := ⟨.program ⟨214⟩, ⟨28139⟩⟩
def transferEvent : Nat := 15497
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15495 .coefficient, .predecessor 1 15496 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15495 .coefficient)
      LeftBound15492.bound (LeftBound15492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15492.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15492.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15496 .coefficient)
      LeftBound11447.bound (LeftBound11447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11451RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11447.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11447.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15492.bound, LeftBound11447.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15492.bound, LeftBound11447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15492.actual selector witness, LeftBound11447.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15497

namespace LeftBound15498
def owner : Owner := ⟨.program ⟨214⟩, ⟨28139⟩⟩
def transferEvent : Nat := 15498
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 15494 .summary, .result 11451 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15494 .summary)
      LeftBound15493.bound (LeftBound15493.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27922⟩⟩) (rawTerms := some (Proof.Events060.exact15494RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11451 .summary)
      LeftBound11450.bound (LeftBound11450.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28138⟩⟩) (rawTerms := some (Proof.Events044.exact11451RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound11450.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15493.bound, LeftBound11450.bound]
def bound : CoeffClass := .finite ⟨11627843036103066759168, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15493.bound, LeftBound11450.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15493.actual selector witness, LeftBound11450.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15498

namespace LeftBound15502
def owner : Owner := ⟨.program ⟨214⟩, ⟨28356⟩⟩
def transferEvent : Nat := 15502
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15500 .coefficient, .predecessor 1 15501 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15500 .coefficient)
      LeftBound15497.bound (LeftBound15497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15499RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15497.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15497.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15501 .coefficient)
      LeftBound10946.bound (LeftBound10946.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10950RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10946.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10946.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15497.bound, LeftBound10946.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15497.bound, LeftBound10946.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15497.actual selector witness, LeftBound10946.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15502

namespace LeftBound15503
def owner : Owner := ⟨.program ⟨214⟩, ⟨28356⟩⟩
def transferEvent : Nat := 15503
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 15499 .summary, .result 10950 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15499 .summary)
      LeftBound15498.bound (LeftBound15498.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28139⟩⟩) (rawTerms := some (Proof.Events060.exact15499RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15498.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10950 .summary)
      LeftBound10949.bound (LeftBound10949.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28355⟩⟩) (rawTerms := some (Proof.Events042.exact10950RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound10949.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15498.bound, LeftBound10949.bound]
def bound : CoeffClass := .finite ⟨12920023572267756019712, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15498.bound, LeftBound10949.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15498.actual selector witness, LeftBound10949.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15503

namespace LeftBound15507
def owner : Owner := ⟨.program ⟨214⟩, ⟨28573⟩⟩
def transferEvent : Nat := 15507
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15505 .coefficient, .predecessor 1 15506 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15505 .coefficient)
      LeftBound15502.bound (LeftBound15502.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15502.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15502.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15506 .coefficient)
      LeftBound10445.bound (LeftBound10445.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10445.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10445.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15502.bound, LeftBound10445.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15502.bound, LeftBound10445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15502.actual selector witness, LeftBound10445.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15507

namespace LeftBound15508
def owner : Owner := ⟨.program ⟨214⟩, ⟨28573⟩⟩
def transferEvent : Nat := 15508
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 15504 .summary, .result 10449 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15504 .summary)
      LeftBound15503.bound (LeftBound15503.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28356⟩⟩) (rawTerms := some (Proof.Events060.exact15504RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15503.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10449 .summary)
      LeftBound10448.bound (LeftBound10448.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28572⟩⟩) (rawTerms := some (Proof.Events040.exact10449RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound10448.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15503.bound, LeftBound10448.bound]
def bound : CoeffClass := .finite ⟨14212226520877465866240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15503.bound, LeftBound10448.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15503.actual selector witness, LeftBound10448.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15508

namespace LeftBound15512
def owner : Owner := ⟨.program ⟨214⟩, ⟨28790⟩⟩
def transferEvent : Nat := 15512
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15510 .coefficient, .predecessor 1 15511 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15510 .coefficient)
      LeftBound15507.bound (LeftBound15507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15509RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15507.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15507.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15511 .coefficient)
      LeftBound9944.bound (LeftBound9944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9944.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9944.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15507.bound, LeftBound9944.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15507.bound, LeftBound9944.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15507.actual selector witness, LeftBound9944.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15512

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
