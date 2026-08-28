import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard083
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard084
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard388

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound57351
def owner : Owner := ⟨.program ⟨214⟩, ⟨15663⟩⟩
def transferEvent : Nat := 57351
def frameStart : Nat := 57292
def rule : BoundRule := .identity (.predecessor 0 57350 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57350 .coefficient)
      LeftBound57348.bound (LeftBound57348.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound57348.derived selector witness)

def rawBound : CoeffClass := LeftBound57348.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57348.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound57348.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound57351

namespace LeftBound57357
def owner : Owner := ⟨.program ⟨214⟩, ⟨15664⟩⟩
def transferEvent : Nat := 57357
def frameStart : Nat := 57292
def rule : BoundRule := .product (.predecessor 0 57355 .coefficient) (.predecessor 1 57356 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57355 .coefficient)
      LeftAuthority57353.bound (LeftAuthority57353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57354RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57353.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57353.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57356 .coefficient)
      LeftBound57351.bound (LeftBound57351.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57352RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57351.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57351.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority57353.bound LeftBound57351.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57353.bound, LeftBound57351.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority57353.actual selector witness) * (LeftBound57351.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57357

namespace LeftBound57365
def owner : Owner := ⟨.program ⟨214⟩, ⟨15665⟩⟩
def transferEvent : Nat := 57365
def frameStart : Nat := 57292
def rule : BoundRule := .sum [.predecessor 0 57363 .coefficient, .predecessor 1 57364 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57363 .coefficient)
      LeftAuthority57361.bound (LeftAuthority57361.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57362RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57361.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57361.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57364 .coefficient)
      LeftBound57357.bound (LeftBound57357.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57359RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57357.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57357.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority57361.bound, LeftBound57357.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57361.bound, LeftBound57357.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority57361.actual selector witness, LeftBound57357.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57365

namespace LeftBound57369
def owner : Owner := ⟨.program ⟨214⟩, ⟨27229⟩⟩
def transferEvent : Nat := 57369
def frameStart : Nat := 57292
def rule : BoundRule := .product (.predecessor 0 57367 .coefficient) (.predecessor 1 57368 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57367 .coefficient)
      LeftBound57365.bound (LeftBound57365.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57366RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57365.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57365.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57368 .coefficient)
      LeftAuthority57342.bound (LeftAuthority57342.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57342.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57342.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound57365.bound LeftAuthority57342.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57365.bound, LeftAuthority57342.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound57365.actual selector witness) * (LeftAuthority57342.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57369

namespace LeftBound57380
def owner : Owner := ⟨.program ⟨214⟩, ⟨15633⟩⟩
def transferEvent : Nat := 57380
def frameStart : Nat := 57292
def rule : BoundRule := .product (.predecessor 0 57378 .coefficient) (.predecessor 1 57379 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57378 .coefficient)
      LeftAuthority57353.bound (LeftAuthority57353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57354RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57353.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57353.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57379 .coefficient)
      LeftAuthority57376.bound (LeftAuthority57376.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57377RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57376.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57376.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority57353.bound LeftAuthority57376.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57353.bound, LeftAuthority57376.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority57353.actual selector witness) * (LeftAuthority57376.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57380

namespace LeftBound57388
def owner : Owner := ⟨.program ⟨214⟩, ⟨15634⟩⟩
def transferEvent : Nat := 57388
def frameStart : Nat := 57292
def rule : BoundRule := .sum [.predecessor 0 57386 .coefficient, .predecessor 1 57387 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57386 .coefficient)
      LeftAuthority57384.bound (LeftAuthority57384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57385RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority57384.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority57384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57387 .coefficient)
      LeftBound57380.bound (LeftBound57380.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57382RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57380.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57380.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority57384.bound, LeftBound57380.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority57384.bound, LeftBound57380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority57384.actual selector witness, LeftBound57380.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57388

namespace LeftBound57392
def owner : Owner := ⟨.program ⟨214⟩, ⟨27233⟩⟩
def transferEvent : Nat := 57392
def frameStart : Nat := 57292
def rule : BoundRule := .sum [.predecessor 0 57390 .coefficient, .predecessor 1 57391 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57390 .coefficient)
      LeftBound57388.bound (LeftBound57388.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57389RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57388.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57388.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57391 .coefficient)
      LeftBound57369.bound (LeftBound57369.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57374RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57369.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57369.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57388.bound, LeftBound57369.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57388.bound, LeftBound57369.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57388.actual selector witness, LeftBound57369.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57392

namespace LeftBound57405
def owner : Owner := ⟨.program ⟨214⟩, ⟨27231⟩⟩
def transferEvent : Nat := 57405
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 57403 .coefficient, .predecessor 1 57404 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57403 .coefficient)
      LeftBound57234.bound (LeftBound57234.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57234.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57234.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57404 .coefficient)
      LeftBound57217.bound (LeftBound57217.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events223.exact57224RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57217.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57217.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57234.bound, LeftBound57217.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57234.bound, LeftBound57217.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57234.actual selector witness, LeftBound57217.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57405

namespace LeftBound57408
def owner : Owner := ⟨.program ⟨214⟩, ⟨27231⟩⟩
def transferEvent : Nat := 57408
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 57402 .summary, .result 57224 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57402 .summary)
      LeftBound57236.bound (LeftBound57236.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20975⟩⟩) (rawTerms := some (Proof.Events224.exact57402RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57236.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57224 .summary)
      LeftBound57219.bound (LeftBound57219.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27230⟩⟩) (rawTerms := some (Proof.Events223.exact57224RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57219.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57236.bound, LeftBound57219.bound]
def bound : CoeffClass := .finite ⟨1291978824159503986688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57236.bound, LeftBound57219.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57236.actual selector witness, LeftBound57219.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57408

namespace LeftBound57432
def owner : Owner := ⟨.program ⟨214⟩, ⟨11138⟩⟩
def transferEvent : Nat := 57432
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 57430 .coefficient) (.predecessor 1 57431 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57430 .coefficient)
      LeftAuthority2659.bound (LeftAuthority2659.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events010.exact2660RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2659.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2659.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57431 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2659.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2659.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2659.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound57432

namespace LeftBound57437
def owner : Owner := ⟨.program ⟨214⟩, ⟨7269⟩⟩
def transferEvent : Nat := 57437
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 57435 .coefficient) (.predecessor 1 57436 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57435 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57436 .coefficient)
      LeftBound13485.bound (LeftBound13485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13485.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13485.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound13485.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound13485.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound13485.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57437

namespace LeftBound57442
def owner : Owner := ⟨.program ⟨214⟩, ⟨11139⟩⟩
def transferEvent : Nat := 57442
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 57440 .coefficient, .predecessor 1 57441 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57440 .coefficient)
      LeftBound57437.bound (LeftBound57437.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57439RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57437.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57437.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57441 .coefficient)
      LeftBound57432.bound (LeftBound57432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57434RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57432.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57432.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57437.bound, LeftBound57432.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57437.bound, LeftBound57432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57437.actual selector witness, LeftBound57432.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57442

namespace LeftBound57446
def owner : Owner := ⟨.program ⟨214⟩, ⟨11140⟩⟩
def transferEvent : Nat := 57446
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 57444 .coefficient, .predecessor 1 57445 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57444 .coefficient)
      LeftBound57442.bound (LeftBound57442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57443RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57442.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57442.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57445 .coefficient)
      LeftBound13477.bound (LeftBound13477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13477.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound57442.bound, LeftBound13477.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57442.bound, LeftBound13477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound57442.actual selector witness, LeftBound13477.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound57446

namespace LeftBound57447
def owner : Owner := ⟨.program ⟨214⟩, ⟨11140⟩⟩
def transferEvent : Nat := 57447
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨89⟩⟩]⟩ [⟨.result 13478 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13478 .coefficient)
      LeftBound13477.bound (LeftBound13477.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨89⟩⟩) (rawTerms := some (Proof.Events052.exact13478RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13477.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound13477.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound13477.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound57447

namespace LeftBound57452
def owner : Owner := ⟨.program ⟨214⟩, ⟨12175⟩⟩
def transferEvent : Nat := 57452
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 57450 .coefficient) (.predecessor 1 57451 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 57450 .coefficient)
      LeftBound57446.bound (LeftBound57446.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57446.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57446.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 57451 .coefficient)
      LeftAuthority2662.bound (LeftAuthority2662.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events010.exact2663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2662.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2662.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound57446.bound LeftAuthority2662.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound57446.bound, LeftAuthority2662.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound57446.actual selector witness) * (LeftAuthority2662.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound57452

namespace LeftBound57453
def owner : Owner := ⟨.program ⟨214⟩, ⟨12175⟩⟩
def transferEvent : Nat := 57453
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨12172⟩⟩], []⟩ [⟨.result 2663 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2663 .coefficient)
      LeftAuthority2662.bound (LeftAuthority2662.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨12172⟩⟩) (rawTerms := some (Proof.Events010.exact2663RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2662.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2662.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority2662.bound []
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2662.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority2662.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound57453

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
