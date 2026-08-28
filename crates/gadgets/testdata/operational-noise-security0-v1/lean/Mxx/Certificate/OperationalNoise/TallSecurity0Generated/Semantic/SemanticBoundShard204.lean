import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard203

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound31306
def owner : Owner := ⟨.program ⟨214⟩, ⟨17357⟩⟩
def transferEvent : Nat := 31306
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31304 .coefficient, .predecessor 1 31305 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31304 .coefficient)
      LeftBound31302.bound (LeftBound31302.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31303RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31302.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31302.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31305 .coefficient)
      LeftAuthority31171.bound (LeftAuthority31171.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events121.exact31172RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31171.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31171.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31302.bound, LeftAuthority31171.bound]
def bound : CoeffClass := .finite ⟨314, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31302.bound, LeftAuthority31171.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31302.actual selector witness, LeftAuthority31171.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31306

namespace LeftBound31310
def owner : Owner := ⟨.program ⟨214⟩, ⟨17358⟩⟩
def transferEvent : Nat := 31310
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31308 .coefficient, .predecessor 1 31309 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31308 .coefficient)
      LeftBound31306.bound (LeftBound31306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31306.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31306.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31309 .coefficient)
      LeftAuthority31148.bound (LeftAuthority31148.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events121.exact31149RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31148.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31148.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31306.bound, LeftAuthority31148.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31306.bound, LeftAuthority31148.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31306.actual selector witness, LeftAuthority31148.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31310

namespace LeftBound31314
def owner : Owner := ⟨.program ⟨214⟩, ⟨17359⟩⟩
def transferEvent : Nat := 31314
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31312 .coefficient, .predecessor 1 31313 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31312 .coefficient)
      LeftBound31310.bound (LeftBound31310.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31311RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31310.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31310.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31313 .coefficient)
      LeftAuthority31125.bound (LeftAuthority31125.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events121.exact31126RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31125.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31125.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31310.bound, LeftAuthority31125.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31310.bound, LeftAuthority31125.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31310.actual selector witness, LeftAuthority31125.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31314

namespace LeftBound31318
def owner : Owner := ⟨.program ⟨214⟩, ⟨17360⟩⟩
def transferEvent : Nat := 31318
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31316 .coefficient, .predecessor 1 31317 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31316 .coefficient)
      LeftBound31314.bound (LeftBound31314.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31315RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31314.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31314.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31317 .coefficient)
      LeftAuthority31102.bound (LeftAuthority31102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events121.exact31103RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31102.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31102.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31314.bound, LeftAuthority31102.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31314.bound, LeftAuthority31102.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31314.actual selector witness, LeftAuthority31102.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31318

namespace LeftBound31322
def owner : Owner := ⟨.program ⟨214⟩, ⟨18380⟩⟩
def transferEvent : Nat := 31322
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31320 .coefficient, .predecessor 1 31321 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31320 .coefficient)
      LeftBound31318.bound (LeftBound31318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31321 .coefficient)
      LeftAuthority31079.bound (LeftAuthority31079.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events121.exact31080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31079.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31079.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31318.bound, LeftAuthority31079.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31318.bound, LeftAuthority31079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31318.actual selector witness, LeftAuthority31079.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31322

namespace LeftBound31326
def owner : Owner := ⟨.program ⟨214⟩, ⟨18381⟩⟩
def transferEvent : Nat := 31326
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31324 .coefficient, .predecessor 1 31325 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31324 .coefficient)
      LeftBound31322.bound (LeftBound31322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31322.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31322.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31325 .coefficient)
      LeftAuthority31056.bound (LeftAuthority31056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events121.exact31057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31056.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31056.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31322.bound, LeftAuthority31056.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31322.bound, LeftAuthority31056.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31322.actual selector witness, LeftAuthority31056.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31326

namespace LeftBound31330
def owner : Owner := ⟨.program ⟨214⟩, ⟨18382⟩⟩
def transferEvent : Nat := 31330
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31328 .coefficient, .predecessor 1 31329 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31328 .coefficient)
      LeftBound31326.bound (LeftBound31326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31327RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31326.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31326.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31329 .coefficient)
      LeftAuthority31033.bound (LeftAuthority31033.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events121.exact31034RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31033.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31033.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31326.bound, LeftAuthority31033.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31326.bound, LeftAuthority31033.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31326.actual selector witness, LeftAuthority31033.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31330

namespace LeftBound31334
def owner : Owner := ⟨.program ⟨214⟩, ⟨18383⟩⟩
def transferEvent : Nat := 31334
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31332 .coefficient, .predecessor 1 31333 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31332 .coefficient)
      LeftBound31330.bound (LeftBound31330.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31331RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31330.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31330.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31333 .coefficient)
      LeftAuthority31010.bound (LeftAuthority31010.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events121.exact31011RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31010.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31010.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31330.bound, LeftAuthority31010.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31330.bound, LeftAuthority31010.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31330.actual selector witness, LeftAuthority31010.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31334

namespace LeftBound31338
def owner : Owner := ⟨.program ⟨214⟩, ⟨18384⟩⟩
def transferEvent : Nat := 31338
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31336 .coefficient, .predecessor 1 31337 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31336 .coefficient)
      LeftBound31334.bound (LeftBound31334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31335RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31334.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31334.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31337 .coefficient)
      LeftAuthority30987.bound (LeftAuthority30987.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events121.exact30988RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority30987.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority30987.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31334.bound, LeftAuthority30987.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31334.bound, LeftAuthority30987.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31334.actual selector witness, LeftAuthority30987.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31338

namespace LeftBound31342
def owner : Owner := ⟨.program ⟨214⟩, ⟨18385⟩⟩
def transferEvent : Nat := 31342
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31340 .coefficient, .predecessor 1 31341 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31340 .coefficient)
      LeftBound31338.bound (LeftBound31338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31339RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31338.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31338.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31341 .coefficient)
      LeftAuthority30964.bound (LeftAuthority30964.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events120.exact30965RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority30964.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority30964.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31338.bound, LeftAuthority30964.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31338.bound, LeftAuthority30964.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31338.actual selector witness, LeftAuthority30964.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31342

namespace LeftBound31346
def owner : Owner := ⟨.program ⟨214⟩, ⟨18386⟩⟩
def transferEvent : Nat := 31346
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31344 .coefficient, .predecessor 1 31345 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31344 .coefficient)
      LeftBound31342.bound (LeftBound31342.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31342.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31342.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31345 .coefficient)
      LeftAuthority30941.bound (LeftAuthority30941.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events120.exact30942RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority30941.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority30941.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31342.bound, LeftAuthority30941.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31342.bound, LeftAuthority30941.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31342.actual selector witness, LeftAuthority30941.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31346

namespace LeftBound31350
def owner : Owner := ⟨.program ⟨214⟩, ⟨18387⟩⟩
def transferEvent : Nat := 31350
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31348 .coefficient, .predecessor 1 31349 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31348 .coefficient)
      LeftBound31346.bound (LeftBound31346.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31347RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31346.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31346.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31349 .coefficient)
      LeftAuthority30918.bound (LeftAuthority30918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events120.exact30919RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority30918.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority30918.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31346.bound, LeftAuthority30918.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31346.bound, LeftAuthority30918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31346.actual selector witness, LeftAuthority30918.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31350

namespace LeftBound31354
def owner : Owner := ⟨.program ⟨214⟩, ⟨18388⟩⟩
def transferEvent : Nat := 31354
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31352 .coefficient, .predecessor 1 31353 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31352 .coefficient)
      LeftBound31350.bound (LeftBound31350.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31351RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31350.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31350.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31353 .coefficient)
      LeftAuthority30895.bound (LeftAuthority30895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events120.exact30896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority30895.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority30895.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31350.bound, LeftAuthority30895.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31350.bound, LeftAuthority30895.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31350.actual selector witness, LeftAuthority30895.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31354

namespace LeftBound31357
def owner : Owner := ⟨.program ⟨214⟩, ⟨18389⟩⟩
def transferEvent : Nat := 31357
def frameStart : Nat := 30853
def rule : BoundRule := .identity (.predecessor 0 31356 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31356 .coefficient)
      LeftBound31354.bound (LeftBound31354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31355RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31354.derived selector witness)

def rawBound : CoeffClass := LeftBound31354.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound31354.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound31357

namespace LeftBound31374
def owner : Owner := ⟨.program ⟨214⟩, ⟨18659⟩⟩
def transferEvent : Nat := 31374
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31372 .coefficient, .predecessor 1 31373 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31372 .coefficient)
      LeftBound31357.bound (LeftBound31357.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound31357.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31373 .coefficient)
      LeftAuthority31370.bound (LeftAuthority31370.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority31370.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31357.bound, LeftAuthority31370.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31357.bound, LeftAuthority31370.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31357.actual selector witness, LeftAuthority31370.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31374

namespace LeftBound31377
def owner : Owner := ⟨.program ⟨214⟩, ⟨18660⟩⟩
def transferEvent : Nat := 31377
def frameStart : Nat := 30853
def rule : BoundRule := .identity (.predecessor 0 31376 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31376 .coefficient)
      LeftBound31374.bound (LeftBound31374.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound31374.derived selector witness)

def rawBound : CoeffClass := LeftBound31374.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31374.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound31374.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound31377

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
