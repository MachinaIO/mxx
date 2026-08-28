import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard597

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound87343
def owner : Owner := ⟨.program ⟨214⟩, ⟨6791⟩⟩
def transferEvent : Nat := 87343
def frameStart : Nat := 87267
def rule : BoundRule := .identity (.predecessor 0 87342 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87342 .coefficient)
      LeftAuthority87330.bound (LeftAuthority87330.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87331RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87330.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87330.derived selector witness)

def rawBound : CoeffClass := LeftAuthority87330.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87330.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority87330.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound87343

namespace LeftBound87347
def owner : Owner := ⟨.program ⟨214⟩, ⟨7839⟩⟩
def transferEvent : Nat := 87347
def frameStart : Nat := 87267
def rule : BoundRule := .product (.predecessor 0 87345 .coefficient) (.predecessor 1 87346 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87345 .coefficient)
      LeftBound87343.bound (LeftBound87343.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87344RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87343.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87343.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87346 .coefficient)
      LeftBound87340.bound (LeftBound87340.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87341RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87340.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87340.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound87343.bound LeftBound87340.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87343.bound, LeftBound87340.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound87343.actual selector witness) * (LeftBound87340.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87347

namespace LeftBound87352
def owner : Owner := ⟨.program ⟨214⟩, ⟨11076⟩⟩
def transferEvent : Nat := 87352
def frameStart : Nat := 87267
def rule : BoundRule := .sum [.predecessor 0 87350 .coefficient, .predecessor 1 87351 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87350 .coefficient)
      LeftBound87347.bound (LeftBound87347.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87347.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87347.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87351 .coefficient)
      LeftBound87326.bound (LeftBound87326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87328RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87326.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87326.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87347.bound, LeftBound87326.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87347.bound, LeftBound87326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87347.actual selector witness, LeftBound87326.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87352

namespace LeftBound87356
def owner : Owner := ⟨.program ⟨214⟩, ⟨25068⟩⟩
def transferEvent : Nat := 87356
def frameStart : Nat := 87267
def rule : BoundRule := .product (.predecessor 0 87354 .coefficient) (.predecessor 1 87355 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87354 .coefficient)
      LeftBound87352.bound (LeftBound87352.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87353RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87352.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87352.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87355 .coefficient)
      LeftAuthority87311.bound (LeftAuthority87311.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87312RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87311.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87311.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound87352.bound LeftAuthority87311.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87352.bound, LeftAuthority87311.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound87352.actual selector witness) * (LeftAuthority87311.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87356

namespace LeftBound87367
def owner : Owner := ⟨.program ⟨214⟩, ⟨15116⟩⟩
def transferEvent : Nat := 87367
def frameStart : Nat := 87267
def rule : BoundRule := .product (.predecessor 0 87365 .coefficient) (.predecessor 1 87366 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87365 .coefficient)
      LeftAuthority87322.bound (LeftAuthority87322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87322.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87322.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87366 .coefficient)
      LeftAuthority87363.bound (LeftAuthority87363.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87363.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87363.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority87322.bound LeftAuthority87363.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87322.bound, LeftAuthority87363.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority87322.actual selector witness) * (LeftAuthority87363.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87367

namespace LeftBound87375
def owner : Owner := ⟨.program ⟨214⟩, ⟨15117⟩⟩
def transferEvent : Nat := 87375
def frameStart : Nat := 87267
def rule : BoundRule := .sum [.predecessor 0 87373 .coefficient, .predecessor 1 87374 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87373 .coefficient)
      LeftAuthority87371.bound (LeftAuthority87371.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87372RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87371.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87371.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87374 .coefficient)
      LeftBound87367.bound (LeftBound87367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87367.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority87371.bound, LeftBound87367.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87371.bound, LeftBound87367.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority87371.actual selector witness, LeftBound87367.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87375

namespace LeftBound87379
def owner : Owner := ⟨.program ⟨214⟩, ⟨25069⟩⟩
def transferEvent : Nat := 87379
def frameStart : Nat := 87267
def rule : BoundRule := .sum [.predecessor 0 87377 .coefficient, .predecessor 1 87378 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87377 .coefficient)
      LeftBound87375.bound (LeftBound87375.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87376RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87375.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87375.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87378 .coefficient)
      LeftBound87356.bound (LeftBound87356.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87356.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87356.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87375.bound, LeftBound87356.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87375.bound, LeftBound87356.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87375.actual selector witness, LeftBound87356.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87379

namespace LeftBound87392
def owner : Owner := ⟨.program ⟨214⟩, ⟨25067⟩⟩
def transferEvent : Nat := 87392
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 87390 .coefficient, .predecessor 1 87391 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87390 .coefficient)
      LeftBound87215.bound (LeftBound87215.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87389RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87215.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87215.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87391 .coefficient)
      LeftBound87198.bound (LeftBound87198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87198.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87198.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87215.bound, LeftBound87198.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87215.bound, LeftBound87198.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87215.actual selector witness, LeftBound87198.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87392

namespace LeftBound87395
def owner : Owner := ⟨.program ⟨214⟩, ⟨25067⟩⟩
def transferEvent : Nat := 87395
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 87389 .summary, .result 87205 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87389 .summary)
      LeftBound87217.bound (LeftBound87217.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19171⟩⟩) (rawTerms := some (Proof.Events341.exact87389RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87217.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87205 .summary)
      LeftBound87200.bound (LeftBound87200.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25066⟩⟩) (rawTerms := some (Proof.Events340.exact87205RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87200.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87217.bound, LeftBound87200.bound]
def bound : CoeffClass := .finite ⟨352017970769920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87217.bound, LeftBound87200.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87217.actual selector witness, LeftBound87200.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87395

namespace LeftBound87399
def owner : Owner := ⟨.program ⟨214⟩, ⟨26783⟩⟩
def transferEvent : Nat := 87399
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 87397 .coefficient) (.predecessor 1 87398 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87397 .coefficient)
      LeftBound87392.bound (LeftBound87392.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87392.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87392.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87398 .coefficient)
      LeftAuthority87120.bound (LeftAuthority87120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87121RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87120.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87120.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound87392.bound LeftAuthority87120.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87392.bound, LeftAuthority87120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound87392.actual selector witness) * (LeftAuthority87120.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87399

namespace LeftBound87400
def owner : Owner := ⟨.program ⟨214⟩, ⟨26783⟩⟩
def transferEvent : Nat := 87400
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26781⟩⟩]⟩ [⟨.result 87121 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87121 .coefficient)
      LeftAuthority87120.bound (LeftAuthority87120.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26781⟩⟩) (rawTerms := some (Proof.Events340.exact87121RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87120.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87120.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority87120.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority87120.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound87400

namespace LeftBound87401
def owner : Owner := ⟨.program ⟨214⟩, ⟨26783⟩⟩
def transferEvent : Nat := 87401
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 87396 .summary) (.transfer 87400) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87396 .summary)
      LeftBound87395.bound (LeftBound87395.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25067⟩⟩) (rawTerms := some (Proof.Events341.exact87396RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87395.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 87400)
      LeftBound87400.bound (LeftBound87400.actual selector witness) := by
  exact .transfer (LeftBound87400.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound87395.bound LeftBound87400.bound
def bound : CoeffClass := .finite ⟨1291911585013138718720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87395.bound, LeftBound87400.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound87395.actual selector witness) * (LeftBound87400.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87401

namespace LeftBound87412
def owner : Owner := ⟨.program ⟨214⟩, ⟨20682⟩⟩
def transferEvent : Nat := 87412
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 87410 .coefficient) (.value (.predecessor 1 87411 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87410 .coefficient)
      LeftAuthority87408.bound (LeftAuthority87408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87408.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87408.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87411 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority87408.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87408.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority87408.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound87412

namespace LeftBound87416
def owner : Owner := ⟨.program ⟨214⟩, ⟨20683⟩⟩
def transferEvent : Nat := 87416
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 87414 .coefficient) (.predecessor 1 87415 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87414 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87415 .coefficient)
      LeftBound87412.bound (LeftBound87412.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87413RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87412.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87412.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound87412.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound87412.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound87412.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87416

namespace LeftBound87417
def owner : Owner := ⟨.program ⟨214⟩, ⟨20683⟩⟩
def transferEvent : Nat := 87417
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20680⟩⟩]⟩ [⟨.result 87409 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87409 .coefficient)
      LeftAuthority87408.bound (LeftAuthority87408.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20680⟩⟩) (rawTerms := some (Proof.Events341.exact87409RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87408.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87408.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority87408.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority87408.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound87417

namespace LeftBound87418
def owner : Owner := ⟨.program ⟨214⟩, ⟨20683⟩⟩
def transferEvent : Nat := 87418
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 87417) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 87417)
      LeftBound87417.bound (LeftBound87417.actual selector witness) := by
  exact .transfer (LeftBound87417.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound87417.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound87417.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound87417.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87418

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
