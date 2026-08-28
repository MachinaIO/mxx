import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard436
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard737

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound107385
def owner : Owner := ⟨.program ⟨214⟩, ⟨30153⟩⟩
def transferEvent : Nat := 107385
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107383 .coefficient, .predecessor 1 107384 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107383 .coefficient)
      LeftBound107380.bound (LeftBound107380.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events419.exact107382RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107380.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107380.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107384 .coefficient)
      LeftBound65065.bound (LeftBound65065.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65126RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65065.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65065.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107380.bound, LeftBound65065.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107380.bound, LeftBound65065.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107380.actual selector witness, LeftBound65065.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107385

namespace LeftBound107386
def owner : Owner := ⟨.program ⟨214⟩, ⟨30153⟩⟩
def transferEvent : Nat := 107386
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107382 .summary, .result 65126 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107382 .summary)
      LeftBound107381.bound (LeftBound107381.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30131⟩⟩) (rawTerms := some (Proof.Events419.exact107382RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107381.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65126 .summary)
      LeftBound65067.bound (LeftBound65067.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30152⟩⟩) (rawTerms := some (Proof.Events254.exact65126RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65067.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107381.bound, LeftBound65067.bound]
def bound : CoeffClass := .finite ⟨12658579328807836837657270299134000471437154975796, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107381.bound, LeftBound65067.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107381.actual selector witness, LeftBound65067.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107386

namespace LeftBound107390
def owner : Owner := ⟨.program ⟨214⟩, ⟨30175⟩⟩
def transferEvent : Nat := 107390
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107388 .coefficient, .predecessor 1 107389 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107388 .coefficient)
      LeftBound107385.bound (LeftBound107385.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events419.exact107387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107385.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107389 .coefficient)
      LeftBound50440.bound (LeftBound50440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50501RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50440.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50440.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107385.bound, LeftBound50440.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107385.bound, LeftBound50440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107385.actual selector witness, LeftBound50440.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107390

namespace LeftBound107391
def owner : Owner := ⟨.program ⟨214⟩, ⟨30175⟩⟩
def transferEvent : Nat := 107391
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107387 .summary, .result 50501 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107387 .summary)
      LeftBound107386.bound (LeftBound107386.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30153⟩⟩) (rawTerms := some (Proof.Events419.exact107387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107386.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50501 .summary)
      LeftBound50442.bound (LeftBound50442.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30174⟩⟩) (rawTerms := some (Proof.Events197.exact50501RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50442.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107386.bound, LeftBound50442.bound]
def bound : CoeffClass := .finite ⟨16878105388500579542037270941219941094189086801972, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107386.bound, LeftBound50442.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107386.actual selector witness, LeftBound50442.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107391

namespace LeftBound107395
def owner : Owner := ⟨.program ⟨214⟩, ⟨30197⟩⟩
def transferEvent : Nat := 107395
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107393 .coefficient, .predecessor 1 107394 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107393 .coefficient)
      LeftBound107390.bound (LeftBound107390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events419.exact107392RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107390.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107390.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107394 .coefficient)
      LeftBound35815.bound (LeftBound35815.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35815.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35815.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107390.bound, LeftBound35815.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107390.bound, LeftBound35815.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107390.actual selector witness, LeftBound35815.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107395

namespace LeftBound107396
def owner : Owner := ⟨.program ⟨214⟩, ⟨30197⟩⟩
def transferEvent : Nat := 107396
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107392 .summary, .result 35876 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107392 .summary)
      LeftBound107391.bound (LeftBound107391.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30175⟩⟩) (rawTerms := some (Proof.Events419.exact107392RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107391.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35876 .summary)
      LeftBound35817.bound (LeftBound35817.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30196⟩⟩) (rawTerms := some (Proof.Events140.exact35876RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35817.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107391.bound, LeftBound35817.bound]
def bound : CoeffClass := .finite ⟨21097631448193322246417271583305881716941018628148, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107391.bound, LeftBound35817.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107391.actual selector witness, LeftBound35817.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107396

namespace LeftBound107400
def owner : Owner := ⟨.program ⟨214⟩, ⟨30219⟩⟩
def transferEvent : Nat := 107400
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107398 .coefficient, .predecessor 1 107399 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107398 .coefficient)
      LeftBound107395.bound (LeftBound107395.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events419.exact107397RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107395.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107395.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107399 .coefficient)
      LeftBound21190.bound (LeftBound21190.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21251RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21190.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21190.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107395.bound, LeftBound21190.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107395.bound, LeftBound21190.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107395.actual selector witness, LeftBound21190.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107400

namespace LeftBound107401
def owner : Owner := ⟨.program ⟨214⟩, ⟨30219⟩⟩
def transferEvent : Nat := 107401
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107397 .summary, .result 21251 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107397 .summary)
      LeftBound107396.bound (LeftBound107396.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30197⟩⟩) (rawTerms := some (Proof.Events419.exact107397RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107396.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21251 .summary)
      LeftBound21192.bound (LeftBound21192.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30218⟩⟩) (rawTerms := some (Proof.Events083.exact21251RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21192.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107396.bound, LeftBound21192.bound]
def bound : CoeffClass := .finite ⟨25317157507886064950797272225391822339692950454324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107396.bound, LeftBound21192.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107396.actual selector witness, LeftBound21192.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107401

namespace LeftBound107405
def owner : Owner := ⟨.program ⟨214⟩, ⟨30220⟩⟩
def transferEvent : Nat := 107405
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107403 .coefficient, .predecessor 1 107404 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107403 .coefficient)
      LeftBound107400.bound (LeftBound107400.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events419.exact107402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107400.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107400.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107404 .coefficient)
      LeftBound6274.bound (LeftBound6274.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6275RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6274.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6274.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107400.bound, LeftBound6274.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107400.bound, LeftBound6274.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107400.actual selector witness, LeftBound6274.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107405

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
