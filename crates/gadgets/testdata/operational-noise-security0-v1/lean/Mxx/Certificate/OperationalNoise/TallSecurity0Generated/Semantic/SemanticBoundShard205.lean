import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard204

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound31383
def owner : Owner := ⟨.program ⟨214⟩, ⟨18661⟩⟩
def transferEvent : Nat := 31383
def frameStart : Nat := 30853
def rule : BoundRule := .product (.predecessor 0 31381 .coefficient) (.predecessor 1 31382 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31381 .coefficient)
      LeftAuthority31379.bound (LeftAuthority31379.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31380RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31379.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31379.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31382 .coefficient)
      LeftBound31377.bound (LeftBound31377.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31378RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31377.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31377.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority31379.bound LeftBound31377.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31379.bound, LeftBound31377.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority31379.actual selector witness) * (LeftBound31377.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31383

namespace LeftBound31459
def owner : Owner := ⟨.program ⟨214⟩, ⟨6795⟩⟩
def transferEvent : Nat := 31459
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31457 .coefficient, .predecessor 1 31458 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31457 .coefficient)
      LeftAuthority31455.bound (LeftAuthority31455.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31456RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31455.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31455.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31458 .coefficient)
      LeftAuthority31452.bound (LeftAuthority31452.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31453RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31452.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31452.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority31455.bound, LeftAuthority31452.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31455.bound, LeftAuthority31452.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority31455.actual selector witness, LeftAuthority31452.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31459

namespace LeftBound31463
def owner : Owner := ⟨.program ⟨214⟩, ⟨6796⟩⟩
def transferEvent : Nat := 31463
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31461 .coefficient, .predecessor 1 31462 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31461 .coefficient)
      LeftBound31459.bound (LeftBound31459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31462 .coefficient)
      LeftAuthority31449.bound (LeftAuthority31449.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31450RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31449.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31449.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31459.bound, LeftAuthority31449.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31459.bound, LeftAuthority31449.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31459.actual selector witness, LeftAuthority31449.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31463

namespace LeftBound31467
def owner : Owner := ⟨.program ⟨214⟩, ⟨6797⟩⟩
def transferEvent : Nat := 31467
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31465 .coefficient, .predecessor 1 31466 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31465 .coefficient)
      LeftBound31463.bound (LeftBound31463.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31464RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31463.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31463.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31466 .coefficient)
      LeftAuthority31446.bound (LeftAuthority31446.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31447RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31446.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31446.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31463.bound, LeftAuthority31446.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31463.bound, LeftAuthority31446.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31463.actual selector witness, LeftAuthority31446.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31467

namespace LeftBound31471
def owner : Owner := ⟨.program ⟨214⟩, ⟨6798⟩⟩
def transferEvent : Nat := 31471
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31469 .coefficient, .predecessor 1 31470 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31469 .coefficient)
      LeftBound31467.bound (LeftBound31467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31467.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31470 .coefficient)
      LeftAuthority31443.bound (LeftAuthority31443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31444RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31443.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31443.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31467.bound, LeftAuthority31443.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31467.bound, LeftAuthority31443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31467.actual selector witness, LeftAuthority31443.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31471

namespace LeftBound31475
def owner : Owner := ⟨.program ⟨214⟩, ⟨6799⟩⟩
def transferEvent : Nat := 31475
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31473 .coefficient, .predecessor 1 31474 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31473 .coefficient)
      LeftBound31471.bound (LeftBound31471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31471.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31474 .coefficient)
      LeftAuthority31440.bound (LeftAuthority31440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31440.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31471.bound, LeftAuthority31440.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31471.bound, LeftAuthority31440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31471.actual selector witness, LeftAuthority31440.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31475

namespace LeftBound31479
def owner : Owner := ⟨.program ⟨214⟩, ⟨6800⟩⟩
def transferEvent : Nat := 31479
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31477 .coefficient, .predecessor 1 31478 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31477 .coefficient)
      LeftBound31475.bound (LeftBound31475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31475.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31478 .coefficient)
      LeftAuthority31437.bound (LeftAuthority31437.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31438RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31437.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31437.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31475.bound, LeftAuthority31437.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31475.bound, LeftAuthority31437.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31475.actual selector witness, LeftAuthority31437.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31479

namespace LeftBound31483
def owner : Owner := ⟨.program ⟨214⟩, ⟨6801⟩⟩
def transferEvent : Nat := 31483
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31481 .coefficient, .predecessor 1 31482 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31481 .coefficient)
      LeftBound31479.bound (LeftBound31479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31479.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31482 .coefficient)
      LeftAuthority31434.bound (LeftAuthority31434.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31435RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31434.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31434.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31479.bound, LeftAuthority31434.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31479.bound, LeftAuthority31434.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31479.actual selector witness, LeftAuthority31434.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31483

namespace LeftBound31487
def owner : Owner := ⟨.program ⟨214⟩, ⟨6802⟩⟩
def transferEvent : Nat := 31487
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31485 .coefficient, .predecessor 1 31486 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31485 .coefficient)
      LeftBound31483.bound (LeftBound31483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31483.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31483.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31486 .coefficient)
      LeftAuthority31431.bound (LeftAuthority31431.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31432RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31431.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31431.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31483.bound, LeftAuthority31431.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31483.bound, LeftAuthority31431.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31483.actual selector witness, LeftAuthority31431.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31487

namespace LeftBound31491
def owner : Owner := ⟨.program ⟨214⟩, ⟨6803⟩⟩
def transferEvent : Nat := 31491
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31489 .coefficient, .predecessor 1 31490 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31489 .coefficient)
      LeftBound31487.bound (LeftBound31487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31487.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31487.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31490 .coefficient)
      LeftAuthority31428.bound (LeftAuthority31428.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31429RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31428.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31428.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31487.bound, LeftAuthority31428.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31487.bound, LeftAuthority31428.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31487.actual selector witness, LeftAuthority31428.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31491

namespace LeftBound31495
def owner : Owner := ⟨.program ⟨214⟩, ⟨6804⟩⟩
def transferEvent : Nat := 31495
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31493 .coefficient, .predecessor 1 31494 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31493 .coefficient)
      LeftBound31491.bound (LeftBound31491.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31492RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31491.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31491.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31494 .coefficient)
      LeftAuthority31425.bound (LeftAuthority31425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31426RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31425.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31425.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31491.bound, LeftAuthority31425.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31491.bound, LeftAuthority31425.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31491.actual selector witness, LeftAuthority31425.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31495

namespace LeftBound31499
def owner : Owner := ⟨.program ⟨214⟩, ⟨6805⟩⟩
def transferEvent : Nat := 31499
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31497 .coefficient, .predecessor 1 31498 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31497 .coefficient)
      LeftBound31495.bound (LeftBound31495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31496RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31495.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31495.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31498 .coefficient)
      LeftAuthority31422.bound (LeftAuthority31422.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31423RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31422.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31422.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31495.bound, LeftAuthority31422.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31495.bound, LeftAuthority31422.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31495.actual selector witness, LeftAuthority31422.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31499

namespace LeftBound31503
def owner : Owner := ⟨.program ⟨214⟩, ⟨6806⟩⟩
def transferEvent : Nat := 31503
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31501 .coefficient, .predecessor 1 31502 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31501 .coefficient)
      LeftBound31499.bound (LeftBound31499.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31499.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31499.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31502 .coefficient)
      LeftAuthority31419.bound (LeftAuthority31419.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31419.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31419.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31499.bound, LeftAuthority31419.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31499.bound, LeftAuthority31419.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31499.actual selector witness, LeftAuthority31419.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31503

namespace LeftBound31507
def owner : Owner := ⟨.program ⟨214⟩, ⟨6807⟩⟩
def transferEvent : Nat := 31507
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31505 .coefficient, .predecessor 1 31506 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31505 .coefficient)
      LeftBound31503.bound (LeftBound31503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31503.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31506 .coefficient)
      LeftAuthority31416.bound (LeftAuthority31416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31417RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31416.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31416.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31503.bound, LeftAuthority31416.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31503.bound, LeftAuthority31416.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31503.actual selector witness, LeftAuthority31416.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31507

namespace LeftBound31511
def owner : Owner := ⟨.program ⟨214⟩, ⟨6808⟩⟩
def transferEvent : Nat := 31511
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31509 .coefficient, .predecessor 1 31510 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31509 .coefficient)
      LeftBound31507.bound (LeftBound31507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31507.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31507.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31510 .coefficient)
      LeftAuthority31413.bound (LeftAuthority31413.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31414RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31413.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31413.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31507.bound, LeftAuthority31413.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31507.bound, LeftAuthority31413.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31507.actual selector witness, LeftAuthority31413.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31511

namespace LeftBound31515
def owner : Owner := ⟨.program ⟨214⟩, ⟨6809⟩⟩
def transferEvent : Nat := 31515
def frameStart : Nat := 30853
def rule : BoundRule := .sum [.predecessor 0 31513 .coefficient, .predecessor 1 31514 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31513 .coefficient)
      LeftBound31511.bound (LeftBound31511.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31511.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31511.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31514 .coefficient)
      LeftAuthority31410.bound (LeftAuthority31410.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31411RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31410.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31410.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31511.bound, LeftAuthority31410.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31511.bound, LeftAuthority31410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound31511.actual selector witness, LeftAuthority31410.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31515

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
