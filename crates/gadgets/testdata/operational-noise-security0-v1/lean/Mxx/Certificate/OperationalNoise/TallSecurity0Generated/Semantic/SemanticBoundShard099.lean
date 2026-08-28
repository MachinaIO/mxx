import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard087
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard091
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard095
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard098

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound15381
def owner : Owner := ⟨.program ⟨214⟩, ⟨14809⟩⟩
def transferEvent : Nat := 15381
def frameStart : Nat := 15342
def rule : BoundRule := .identity (.predecessor 0 15380 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15380 .coefficient)
      LeftAuthority15378.bound (LeftAuthority15378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15379RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15378.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15378.derived selector witness)

def rawBound : CoeffClass := LeftAuthority15378.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority15378.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound15381

namespace LeftBound15398
def owner : Owner := ⟨.program ⟨214⟩, ⟨14848⟩⟩
def transferEvent : Nat := 15398
def frameStart : Nat := 15342
def rule : BoundRule := .sum [.predecessor 0 15396 .coefficient, .predecessor 1 15397 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15396 .coefficient)
      LeftBound15381.bound (LeftBound15381.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound15381.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15397 .coefficient)
      LeftAuthority15394.bound (LeftAuthority15394.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority15394.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15381.bound, LeftAuthority15394.bound]
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15381.bound, LeftAuthority15394.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15381.actual selector witness, LeftAuthority15394.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15398

namespace LeftBound15401
def owner : Owner := ⟨.program ⟨214⟩, ⟨14849⟩⟩
def transferEvent : Nat := 15401
def frameStart : Nat := 15342
def rule : BoundRule := .identity (.predecessor 0 15400 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15400 .coefficient)
      LeftBound15398.bound (LeftBound15398.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound15398.derived selector witness)

def rawBound : CoeffClass := LeftBound15398.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15398.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound15398.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound15401

namespace LeftBound15407
def owner : Owner := ⟨.program ⟨214⟩, ⟨14850⟩⟩
def transferEvent : Nat := 15407
def frameStart : Nat := 15342
def rule : BoundRule := .product (.predecessor 0 15405 .coefficient) (.predecessor 1 15406 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15405 .coefficient)
      LeftAuthority15403.bound (LeftAuthority15403.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15404RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15403.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15403.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15406 .coefficient)
      LeftBound15401.bound (LeftBound15401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15401.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15401.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority15403.bound LeftBound15401.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15403.bound, LeftBound15401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority15403.actual selector witness) * (LeftBound15401.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15407

namespace LeftBound15415
def owner : Owner := ⟨.program ⟨214⟩, ⟨14851⟩⟩
def transferEvent : Nat := 15415
def frameStart : Nat := 15342
def rule : BoundRule := .sum [.predecessor 0 15413 .coefficient, .predecessor 1 15414 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15413 .coefficient)
      LeftAuthority15411.bound (LeftAuthority15411.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15412RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15411.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15411.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15414 .coefficient)
      LeftBound15407.bound (LeftBound15407.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15407.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15407.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority15411.bound, LeftBound15407.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15411.bound, LeftBound15407.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority15411.actual selector witness, LeftBound15407.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15415

namespace LeftBound15419
def owner : Owner := ⟨.program ⟨214⟩, ⟨26407⟩⟩
def transferEvent : Nat := 15419
def frameStart : Nat := 15342
def rule : BoundRule := .product (.predecessor 0 15417 .coefficient) (.predecessor 1 15418 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15417 .coefficient)
      LeftBound15415.bound (LeftBound15415.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15416RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15415.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15415.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15418 .coefficient)
      LeftAuthority15392.bound (LeftAuthority15392.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15393RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15392.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15392.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound15415.bound LeftAuthority15392.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15415.bound, LeftAuthority15392.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound15415.actual selector witness) * (LeftAuthority15392.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15419

namespace LeftBound15430
def owner : Owner := ⟨.program ⟨214⟩, ⟨15278⟩⟩
def transferEvent : Nat := 15430
def frameStart : Nat := 15342
def rule : BoundRule := .product (.predecessor 0 15428 .coefficient) (.predecessor 1 15429 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15428 .coefficient)
      LeftAuthority15403.bound (LeftAuthority15403.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15404RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15403.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15403.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15429 .coefficient)
      LeftAuthority15426.bound (LeftAuthority15426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15427RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15426.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15426.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority15403.bound LeftAuthority15426.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15403.bound, LeftAuthority15426.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority15403.actual selector witness) * (LeftAuthority15426.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15430

namespace LeftBound15438
def owner : Owner := ⟨.program ⟨214⟩, ⟨15279⟩⟩
def transferEvent : Nat := 15438
def frameStart : Nat := 15342
def rule : BoundRule := .sum [.predecessor 0 15436 .coefficient, .predecessor 1 15437 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15436 .coefficient)
      LeftAuthority15434.bound (LeftAuthority15434.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15435RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15434.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15434.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15437 .coefficient)
      LeftBound15430.bound (LeftBound15430.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15432RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15430.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15430.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority15434.bound, LeftBound15430.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15434.bound, LeftBound15430.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority15434.actual selector witness, LeftBound15430.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15438

namespace LeftBound15442
def owner : Owner := ⟨.program ⟨214⟩, ⟨26410⟩⟩
def transferEvent : Nat := 15442
def frameStart : Nat := 15342
def rule : BoundRule := .sum [.predecessor 0 15440 .coefficient, .predecessor 1 15441 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15440 .coefficient)
      LeftBound15438.bound (LeftBound15438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15439RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15438.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15438.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15441 .coefficient)
      LeftBound15419.bound (LeftBound15419.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15424RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15419.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15419.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15438.bound, LeftBound15419.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15438.bound, LeftBound15419.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15438.actual selector witness, LeftBound15419.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15442

namespace LeftBound15455
def owner : Owner := ⟨.program ⟨214⟩, ⟨26409⟩⟩
def transferEvent : Nat := 15455
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15453 .coefficient, .predecessor 1 15454 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15453 .coefficient)
      LeftBound15284.bound (LeftBound15284.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15452RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15284.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15284.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15454 .coefficient)
      LeftBound15267.bound (LeftBound15267.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15274RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15267.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15267.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15284.bound, LeftBound15267.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15284.bound, LeftBound15267.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15284.actual selector witness, LeftBound15267.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15455

namespace LeftBound15458
def owner : Owner := ⟨.program ⟨214⟩, ⟨26409⟩⟩
def transferEvent : Nat := 15458
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 15452 .summary, .result 15274 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15452 .summary)
      LeftBound15286.bound (LeftBound15286.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20411⟩⟩) (rawTerms := some (Proof.Events060.exact15452RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15286.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15274 .summary)
      LeftBound15269.bound (LeftBound15269.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26408⟩⟩) (rawTerms := some (Proof.Events059.exact15274RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15269.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15286.bound, LeftBound15269.bound]
def bound : CoeffClass := .finite ⟨1291889174379421642752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15286.bound, LeftBound15269.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15286.actual selector witness, LeftBound15269.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15458

namespace LeftBound15462
def owner : Owner := ⟨.program ⟨214⟩, ⟨26620⟩⟩
def transferEvent : Nat := 15462
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15460 .coefficient, .predecessor 1 15461 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15460 .coefficient)
      LeftBound15455.bound (LeftBound15455.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15459RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15455.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15455.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15461 .coefficient)
      LeftBound14954.bound (LeftBound14954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14958RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14954.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14954.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15455.bound, LeftBound14954.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15455.bound, LeftBound14954.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15455.actual selector witness, LeftBound14954.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15462

namespace LeftBound15463
def owner : Owner := ⟨.program ⟨214⟩, ⟨26620⟩⟩
def transferEvent : Nat := 15463
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 15459 .summary, .result 14958 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15459 .summary)
      LeftBound15458.bound (LeftBound15458.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26409⟩⟩) (rawTerms := some (Proof.Events060.exact15459RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15458.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14958 .summary)
      LeftBound14957.bound (LeftBound14957.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26619⟩⟩) (rawTerms := some (Proof.Events058.exact14958RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound14957.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15458.bound, LeftBound14957.bound]
def bound : CoeffClass := .finite ⟨2583789554981353578496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15458.bound, LeftBound14957.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15458.actual selector witness, LeftBound14957.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15463

namespace LeftBound15467
def owner : Owner := ⟨.program ⟨214⟩, ⟨26837⟩⟩
def transferEvent : Nat := 15467
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15465 .coefficient, .predecessor 1 15466 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15465 .coefficient)
      LeftBound15462.bound (LeftBound15462.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15464RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15462.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15462.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15466 .coefficient)
      LeftBound14453.bound (LeftBound14453.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14453.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14453.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15462.bound, LeftBound14453.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15462.bound, LeftBound14453.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15462.actual selector witness, LeftBound14453.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15467

namespace LeftBound15468
def owner : Owner := ⟨.program ⟨214⟩, ⟨26837⟩⟩
def transferEvent : Nat := 15468
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 15464 .summary, .result 14457 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15464 .summary)
      LeftBound15463.bound (LeftBound15463.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26620⟩⟩) (rawTerms := some (Proof.Events060.exact15464RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15463.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14457 .summary)
      LeftBound14456.bound (LeftBound14456.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26836⟩⟩) (rawTerms := some (Proof.Events056.exact14457RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound14456.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15463.bound, LeftBound14456.bound]
def bound : CoeffClass := .finite ⟨3875701141805795807232, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15463.bound, LeftBound14456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15463.actual selector witness, LeftBound14456.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15468

namespace LeftBound15472
def owner : Owner := ⟨.program ⟨214⟩, ⟨27054⟩⟩
def transferEvent : Nat := 15472
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15470 .coefficient, .predecessor 1 15471 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15470 .coefficient)
      LeftBound15467.bound (LeftBound15467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15469RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15467.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15471 .coefficient)
      LeftBound13952.bound (LeftBound13952.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13956RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13952.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13952.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15467.bound, LeftBound13952.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15467.bound, LeftBound13952.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15467.actual selector witness, LeftBound13952.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15472

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
