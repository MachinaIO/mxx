import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard345
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard349
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard353
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard356
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard360
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard364
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard367
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard371
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard375
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard404

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound59376
def owner : Owner := ⟨.program ⟨214⟩, ⟨28100⟩⟩
def transferEvent : Nat := 59376
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 59372 .summary, .result 55481 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59372 .summary)
      LeftBound59371.bound (LeftBound59371.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27883⟩⟩) (rawTerms := some (Proof.Events231.exact59372RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59371.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55481 .summary)
      LeftBound55480.bound (LeftBound55480.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28099⟩⟩) (rawTerms := some (Proof.Events216.exact55481RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55480.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59371.bound, LeftBound55480.bound]
def bound : CoeffClass := .finite ⟨11627843036103066759168, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59371.bound, LeftBound55480.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59371.actual selector witness, LeftBound55480.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59376

namespace LeftBound59380
def owner : Owner := ⟨.program ⟨214⟩, ⟨28317⟩⟩
def transferEvent : Nat := 59380
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 59378 .coefficient, .predecessor 1 59379 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59378 .coefficient)
      LeftBound59375.bound (LeftBound59375.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59377RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59375.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59375.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59379 .coefficient)
      LeftBound54995.bound (LeftBound54995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54999RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54995.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54995.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59375.bound, LeftBound54995.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59375.bound, LeftBound54995.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59375.actual selector witness, LeftBound54995.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59380

namespace LeftBound59381
def owner : Owner := ⟨.program ⟨214⟩, ⟨28317⟩⟩
def transferEvent : Nat := 59381
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 59377 .summary, .result 54999 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59377 .summary)
      LeftBound59376.bound (LeftBound59376.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28100⟩⟩) (rawTerms := some (Proof.Events231.exact59377RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59376.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54999 .summary)
      LeftBound54998.bound (LeftBound54998.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28316⟩⟩) (rawTerms := some (Proof.Events214.exact54999RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54998.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59376.bound, LeftBound54998.bound]
def bound : CoeffClass := .finite ⟨12920023572267756019712, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59376.bound, LeftBound54998.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59376.actual selector witness, LeftBound54998.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59381

namespace LeftBound59385
def owner : Owner := ⟨.program ⟨214⟩, ⟨28534⟩⟩
def transferEvent : Nat := 59385
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 59383 .coefficient, .predecessor 1 59384 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59383 .coefficient)
      LeftBound59380.bound (LeftBound59380.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59382RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59380.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59380.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59384 .coefficient)
      LeftBound54513.bound (LeftBound54513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54513.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54513.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59380.bound, LeftBound54513.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59380.bound, LeftBound54513.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59380.actual selector witness, LeftBound54513.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59385

namespace LeftBound59386
def owner : Owner := ⟨.program ⟨214⟩, ⟨28534⟩⟩
def transferEvent : Nat := 59386
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 59382 .summary, .result 54517 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59382 .summary)
      LeftBound59381.bound (LeftBound59381.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28317⟩⟩) (rawTerms := some (Proof.Events231.exact59382RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59381.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54517 .summary)
      LeftBound54516.bound (LeftBound54516.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28533⟩⟩) (rawTerms := some (Proof.Events212.exact54517RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54516.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59381.bound, LeftBound54516.bound]
def bound : CoeffClass := .finite ⟨14212226520877465866240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59381.bound, LeftBound54516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59381.actual selector witness, LeftBound54516.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59386

namespace LeftBound59390
def owner : Owner := ⟨.program ⟨214⟩, ⟨28751⟩⟩
def transferEvent : Nat := 59390
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 59388 .coefficient, .predecessor 1 59389 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59388 .coefficient)
      LeftBound59385.bound (LeftBound59385.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59385.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59389 .coefficient)
      LeftBound54031.bound (LeftBound54031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54035RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54031.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54031.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59385.bound, LeftBound54031.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59385.bound, LeftBound54031.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59385.actual selector witness, LeftBound54031.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59390

namespace LeftBound59391
def owner : Owner := ⟨.program ⟨214⟩, ⟨28751⟩⟩
def transferEvent : Nat := 59391
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 59387 .summary, .result 54035 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59387 .summary)
      LeftBound59386.bound (LeftBound59386.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28534⟩⟩) (rawTerms := some (Proof.Events231.exact59387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59386.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54035 .summary)
      LeftBound54034.bound (LeftBound54034.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28750⟩⟩) (rawTerms := some (Proof.Events211.exact54035RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54034.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59386.bound, LeftBound54034.bound]
def bound : CoeffClass := .finite ⟨15504496706822237470720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59386.bound, LeftBound54034.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59386.actual selector witness, LeftBound54034.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59391

namespace LeftBound59395
def owner : Owner := ⟨.program ⟨214⟩, ⟨28968⟩⟩
def transferEvent : Nat := 59395
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 59393 .coefficient, .predecessor 1 59394 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59393 .coefficient)
      LeftBound59390.bound (LeftBound59390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59392RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59390.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59390.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59394 .coefficient)
      LeftBound53549.bound (LeftBound53549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53549.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53549.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59390.bound, LeftBound53549.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59390.bound, LeftBound53549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59390.actual selector witness, LeftBound53549.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59395

namespace LeftBound59396
def owner : Owner := ⟨.program ⟨214⟩, ⟨28968⟩⟩
def transferEvent : Nat := 59396
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 59392 .summary, .result 53553 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59392 .summary)
      LeftBound59391.bound (LeftBound59391.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28751⟩⟩) (rawTerms := some (Proof.Events232.exact59392RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59391.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53553 .summary)
      LeftBound53552.bound (LeftBound53552.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28967⟩⟩) (rawTerms := some (Proof.Events209.exact53553RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53552.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59391.bound, LeftBound53552.bound]
def bound : CoeffClass := .finite ⟨16796811717657050247168, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59391.bound, LeftBound53552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59391.actual selector witness, LeftBound53552.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59396

namespace LeftBound59400
def owner : Owner := ⟨.program ⟨214⟩, ⟨29185⟩⟩
def transferEvent : Nat := 59400
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 59398 .coefficient, .predecessor 1 59399 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59398 .coefficient)
      LeftBound59395.bound (LeftBound59395.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59397RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59395.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59395.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59399 .coefficient)
      LeftBound53067.bound (LeftBound53067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events207.exact53071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53067.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59395.bound, LeftBound53067.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59395.bound, LeftBound53067.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59395.actual selector witness, LeftBound53067.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59400

namespace LeftBound59401
def owner : Owner := ⟨.program ⟨214⟩, ⟨29185⟩⟩
def transferEvent : Nat := 59401
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 59397 .summary, .result 53071 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59397 .summary)
      LeftBound59396.bound (LeftBound59396.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28968⟩⟩) (rawTerms := some (Proof.Events232.exact59397RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59396.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53071 .summary)
      LeftBound53070.bound (LeftBound53070.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29184⟩⟩) (rawTerms := some (Proof.Events207.exact53071RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53070.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59396.bound, LeftBound53070.bound]
def bound : CoeffClass := .finite ⟨18089149140936883609600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59396.bound, LeftBound53070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59396.actual selector witness, LeftBound53070.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59401

namespace LeftBound59405
def owner : Owner := ⟨.program ⟨214⟩, ⟨29402⟩⟩
def transferEvent : Nat := 59405
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 59403 .coefficient, .predecessor 1 59404 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59403 .coefficient)
      LeftBound59400.bound (LeftBound59400.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59400.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59400.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59404 .coefficient)
      LeftBound52585.bound (LeftBound52585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52585.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52585.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59400.bound, LeftBound52585.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59400.bound, LeftBound52585.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59400.actual selector witness, LeftBound52585.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59405

namespace LeftBound59406
def owner : Owner := ⟨.program ⟨214⟩, ⟨29402⟩⟩
def transferEvent : Nat := 59406
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 59402 .summary, .result 52589 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59402 .summary)
      LeftBound59401.bound (LeftBound59401.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29185⟩⟩) (rawTerms := some (Proof.Events232.exact59402RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59401.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 52589 .summary)
      LeftBound52588.bound (LeftBound52588.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29401⟩⟩) (rawTerms := some (Proof.Events205.exact52589RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound52588.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59401.bound, LeftBound52588.bound]
def bound : CoeffClass := .finite ⟨19381531389106758144000, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59401.bound, LeftBound52588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59401.actual selector witness, LeftBound52588.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59406

namespace LeftBound59410
def owner : Owner := ⟨.program ⟨214⟩, ⟨29619⟩⟩
def transferEvent : Nat := 59410
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 59408 .coefficient, .predecessor 1 59409 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59408 .coefficient)
      LeftBound59405.bound (LeftBound59405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59407RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59405.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59405.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59409 .coefficient)
      LeftBound52103.bound (LeftBound52103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52107RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52103.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52103.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59405.bound, LeftBound52103.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59405.bound, LeftBound52103.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59405.actual selector witness, LeftBound52103.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59410

namespace LeftBound59411
def owner : Owner := ⟨.program ⟨214⟩, ⟨29619⟩⟩
def transferEvent : Nat := 59411
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 59407 .summary, .result 52107 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59407 .summary)
      LeftBound59406.bound (LeftBound59406.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29402⟩⟩) (rawTerms := some (Proof.Events232.exact59407RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59406.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 52107 .summary)
      LeftBound52106.bound (LeftBound52106.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29618⟩⟩) (rawTerms := some (Proof.Events203.exact52107RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound52106.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59406.bound, LeftBound52106.bound]
def bound : CoeffClass := .finite ⟨20673980874611694436352, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59406.bound, LeftBound52106.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59406.actual selector witness, LeftBound52106.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59411

namespace LeftBound59415
def owner : Owner := ⟨.program ⟨214⟩, ⟨29836⟩⟩
def transferEvent : Nat := 59415
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 59413 .coefficient, .predecessor 1 59414 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59413 .coefficient)
      LeftBound59410.bound (LeftBound59410.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59412RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59410.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59410.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59414 .coefficient)
      LeftBound51621.bound (LeftBound51621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51621.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51621.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59410.bound, LeftBound51621.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59410.bound, LeftBound51621.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59410.actual selector witness, LeftBound51621.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59415

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
