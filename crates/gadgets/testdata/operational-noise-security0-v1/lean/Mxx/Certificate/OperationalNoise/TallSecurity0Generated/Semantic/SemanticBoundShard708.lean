import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound103352
def owner : Owner := ⟨.program ⟨214⟩, ⟨15301⟩⟩
def transferEvent : Nat := 103352
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103350 .coefficient, .predecessor 1 103351 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103350 .coefficient)
      LeftAuthority103348.bound (LeftAuthority103348.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103348.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103348.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103351 .coefficient)
      LeftAuthority103325.bound (LeftAuthority103325.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103326RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103325.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103325.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority103348.bound, LeftAuthority103325.bound]
def bound : CoeffClass := .finite ⟨91, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority103348.bound, LeftAuthority103325.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority103348.actual selector witness, LeftAuthority103325.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103352

namespace LeftBound103356
def owner : Owner := ⟨.program ⟨214⟩, ⟨15357⟩⟩
def transferEvent : Nat := 103356
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103354 .coefficient, .predecessor 1 103355 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103354 .coefficient)
      LeftBound103352.bound (LeftBound103352.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103353RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103352.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103352.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103355 .coefficient)
      LeftAuthority103302.bound (LeftAuthority103302.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103303RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103302.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103302.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103352.bound, LeftAuthority103302.bound]
def bound : CoeffClass := .finite ⟨142, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103352.bound, LeftAuthority103302.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103352.actual selector witness, LeftAuthority103302.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103356

namespace LeftBound103360
def owner : Owner := ⟨.program ⟨214⟩, ⟨17303⟩⟩
def transferEvent : Nat := 103360
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103358 .coefficient, .predecessor 1 103359 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103358 .coefficient)
      LeftBound103356.bound (LeftBound103356.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103357RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103356.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103356.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103359 .coefficient)
      LeftAuthority103279.bound (LeftAuthority103279.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103279.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103279.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103356.bound, LeftAuthority103279.bound]
def bound : CoeffClass := .finite ⟨197, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103356.bound, LeftAuthority103279.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103356.actual selector witness, LeftAuthority103279.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103360

namespace LeftBound103364
def owner : Owner := ⟨.program ⟨214⟩, ⟨17304⟩⟩
def transferEvent : Nat := 103364
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103362 .coefficient, .predecessor 1 103363 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103362 .coefficient)
      LeftBound103360.bound (LeftBound103360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103360.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103360.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103363 .coefficient)
      LeftAuthority103256.bound (LeftAuthority103256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103257RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103256.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103256.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103360.bound, LeftAuthority103256.bound]
def bound : CoeffClass := .finite ⟨255, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103360.bound, LeftAuthority103256.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103360.actual selector witness, LeftAuthority103256.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103364

namespace LeftBound103368
def owner : Owner := ⟨.program ⟨214⟩, ⟨17305⟩⟩
def transferEvent : Nat := 103368
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103366 .coefficient, .predecessor 1 103367 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103366 .coefficient)
      LeftBound103364.bound (LeftBound103364.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103365RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103364.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103364.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103367 .coefficient)
      LeftAuthority103233.bound (LeftAuthority103233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103234RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103233.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103233.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103364.bound, LeftAuthority103233.bound]
def bound : CoeffClass := .finite ⟨314, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103364.bound, LeftAuthority103233.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103364.actual selector witness, LeftAuthority103233.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103368

namespace LeftBound103372
def owner : Owner := ⟨.program ⟨214⟩, ⟨17306⟩⟩
def transferEvent : Nat := 103372
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103370 .coefficient, .predecessor 1 103371 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103370 .coefficient)
      LeftBound103368.bound (LeftBound103368.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103368.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103371 .coefficient)
      LeftAuthority103210.bound (LeftAuthority103210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103210.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103210.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103368.bound, LeftAuthority103210.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103368.bound, LeftAuthority103210.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103368.actual selector witness, LeftAuthority103210.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103372

namespace LeftBound103376
def owner : Owner := ⟨.program ⟨214⟩, ⟨17307⟩⟩
def transferEvent : Nat := 103376
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103374 .coefficient, .predecessor 1 103375 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103374 .coefficient)
      LeftBound103372.bound (LeftBound103372.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103373RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103372.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103372.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103375 .coefficient)
      LeftAuthority103187.bound (LeftAuthority103187.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103187.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103187.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103372.bound, LeftAuthority103187.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103372.bound, LeftAuthority103187.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103372.actual selector witness, LeftAuthority103187.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103376

namespace LeftBound103380
def owner : Owner := ⟨.program ⟨214⟩, ⟨17308⟩⟩
def transferEvent : Nat := 103380
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103378 .coefficient, .predecessor 1 103379 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103378 .coefficient)
      LeftBound103376.bound (LeftBound103376.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103377RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103376.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103376.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103379 .coefficient)
      LeftAuthority103164.bound (LeftAuthority103164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events402.exact103165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103164.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103164.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103376.bound, LeftAuthority103164.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103376.bound, LeftAuthority103164.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103376.actual selector witness, LeftAuthority103164.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103380

namespace LeftBound103384
def owner : Owner := ⟨.program ⟨214⟩, ⟨18304⟩⟩
def transferEvent : Nat := 103384
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103382 .coefficient, .predecessor 1 103383 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103382 .coefficient)
      LeftBound103380.bound (LeftBound103380.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103381RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103380.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103380.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103383 .coefficient)
      LeftAuthority103141.bound (LeftAuthority103141.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events402.exact103142RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103141.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103141.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103380.bound, LeftAuthority103141.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103380.bound, LeftAuthority103141.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103380.actual selector witness, LeftAuthority103141.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103384

namespace LeftBound103388
def owner : Owner := ⟨.program ⟨214⟩, ⟨18305⟩⟩
def transferEvent : Nat := 103388
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103386 .coefficient, .predecessor 1 103387 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103386 .coefficient)
      LeftBound103384.bound (LeftBound103384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103385RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103387 .coefficient)
      LeftAuthority103118.bound (LeftAuthority103118.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events402.exact103119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103118.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103118.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103384.bound, LeftAuthority103118.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103384.bound, LeftAuthority103118.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103384.actual selector witness, LeftAuthority103118.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103388

namespace LeftBound103392
def owner : Owner := ⟨.program ⟨214⟩, ⟨18306⟩⟩
def transferEvent : Nat := 103392
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103390 .coefficient, .predecessor 1 103391 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103390 .coefficient)
      LeftBound103388.bound (LeftBound103388.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103389RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103388.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103388.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103391 .coefficient)
      LeftAuthority103095.bound (LeftAuthority103095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events402.exact103096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103095.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103095.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103388.bound, LeftAuthority103095.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103388.bound, LeftAuthority103095.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103388.actual selector witness, LeftAuthority103095.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103392

namespace LeftBound103396
def owner : Owner := ⟨.program ⟨214⟩, ⟨18307⟩⟩
def transferEvent : Nat := 103396
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103394 .coefficient, .predecessor 1 103395 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103394 .coefficient)
      LeftBound103392.bound (LeftBound103392.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103393RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103392.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103392.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103395 .coefficient)
      LeftAuthority103072.bound (LeftAuthority103072.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events402.exact103073RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103072.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103072.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103392.bound, LeftAuthority103072.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103392.bound, LeftAuthority103072.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103392.actual selector witness, LeftAuthority103072.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103396

namespace LeftBound103400
def owner : Owner := ⟨.program ⟨214⟩, ⟨18308⟩⟩
def transferEvent : Nat := 103400
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103398 .coefficient, .predecessor 1 103399 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103398 .coefficient)
      LeftBound103396.bound (LeftBound103396.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103397RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103396.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103396.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103399 .coefficient)
      LeftAuthority103049.bound (LeftAuthority103049.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events402.exact103050RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103049.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103049.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103396.bound, LeftAuthority103049.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103396.bound, LeftAuthority103049.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103396.actual selector witness, LeftAuthority103049.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103400

namespace LeftBound103404
def owner : Owner := ⟨.program ⟨214⟩, ⟨18309⟩⟩
def transferEvent : Nat := 103404
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103402 .coefficient, .predecessor 1 103403 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103402 .coefficient)
      LeftBound103400.bound (LeftBound103400.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103401RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103400.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103400.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103403 .coefficient)
      LeftAuthority103026.bound (LeftAuthority103026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events402.exact103027RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103026.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103026.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103400.bound, LeftAuthority103026.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103400.bound, LeftAuthority103026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103400.actual selector witness, LeftAuthority103026.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103404

namespace LeftBound103408
def owner : Owner := ⟨.program ⟨214⟩, ⟨18310⟩⟩
def transferEvent : Nat := 103408
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103406 .coefficient, .predecessor 1 103407 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103406 .coefficient)
      LeftBound103404.bound (LeftBound103404.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103405RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103404.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103404.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103407 .coefficient)
      LeftAuthority103003.bound (LeftAuthority103003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events402.exact103004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103003.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103003.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103404.bound, LeftAuthority103003.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103404.bound, LeftAuthority103003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103404.actual selector witness, LeftAuthority103003.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103408

namespace LeftBound103412
def owner : Owner := ⟨.program ⟨214⟩, ⟨18311⟩⟩
def transferEvent : Nat := 103412
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103410 .coefficient, .predecessor 1 103411 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103410 .coefficient)
      LeftBound103408.bound (LeftBound103408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103408.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103408.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103411 .coefficient)
      LeftAuthority102980.bound (LeftAuthority102980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events402.exact102981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102980.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102980.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103408.bound, LeftAuthority102980.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103408.bound, LeftAuthority102980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103408.actual selector witness, LeftAuthority102980.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103412

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
