import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard507
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard508
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard509

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound75374
def owner : Owner := ⟨.program ⟨214⟩, ⟨6805⟩⟩
def transferEvent : Nat := 75374
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75372 .coefficient, .predecessor 1 75373 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75372 .coefficient)
      LeftBound75370.bound (LeftBound75370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75371RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75370.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75370.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75373 .coefficient)
      LeftAuthority75297.bound (LeftAuthority75297.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75298RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75297.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75297.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75370.bound, LeftAuthority75297.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75370.bound, LeftAuthority75297.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75370.actual selector witness, LeftAuthority75297.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75374

namespace LeftBound75378
def owner : Owner := ⟨.program ⟨214⟩, ⟨6806⟩⟩
def transferEvent : Nat := 75378
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75376 .coefficient, .predecessor 1 75377 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75376 .coefficient)
      LeftBound75374.bound (LeftBound75374.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75375RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75374.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75374.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75377 .coefficient)
      LeftAuthority75294.bound (LeftAuthority75294.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75294.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75294.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75374.bound, LeftAuthority75294.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75374.bound, LeftAuthority75294.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75374.actual selector witness, LeftAuthority75294.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75378

namespace LeftBound75382
def owner : Owner := ⟨.program ⟨214⟩, ⟨6807⟩⟩
def transferEvent : Nat := 75382
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75380 .coefficient, .predecessor 1 75381 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75380 .coefficient)
      LeftBound75378.bound (LeftBound75378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75379RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75378.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75378.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75381 .coefficient)
      LeftAuthority75291.bound (LeftAuthority75291.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75292RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75291.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75291.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75378.bound, LeftAuthority75291.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75378.bound, LeftAuthority75291.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75378.actual selector witness, LeftAuthority75291.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75382

namespace LeftBound75386
def owner : Owner := ⟨.program ⟨214⟩, ⟨6808⟩⟩
def transferEvent : Nat := 75386
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75384 .coefficient, .predecessor 1 75385 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75384 .coefficient)
      LeftBound75382.bound (LeftBound75382.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75383RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75382.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75382.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75385 .coefficient)
      LeftAuthority75288.bound (LeftAuthority75288.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75289RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75288.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75288.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75382.bound, LeftAuthority75288.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75382.bound, LeftAuthority75288.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75382.actual selector witness, LeftAuthority75288.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75386

namespace LeftBound75390
def owner : Owner := ⟨.program ⟨214⟩, ⟨6809⟩⟩
def transferEvent : Nat := 75390
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75388 .coefficient, .predecessor 1 75389 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75388 .coefficient)
      LeftBound75386.bound (LeftBound75386.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75386.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75386.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75389 .coefficient)
      LeftAuthority75285.bound (LeftAuthority75285.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75286RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75285.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75285.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75386.bound, LeftAuthority75285.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75386.bound, LeftAuthority75285.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75386.actual selector witness, LeftAuthority75285.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75390

namespace LeftBound75394
def owner : Owner := ⟨.program ⟨214⟩, ⟨6810⟩⟩
def transferEvent : Nat := 75394
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75392 .coefficient, .predecessor 1 75393 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75392 .coefficient)
      LeftBound75390.bound (LeftBound75390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75391RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75390.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75390.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75393 .coefficient)
      LeftAuthority75282.bound (LeftAuthority75282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75282.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75282.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75390.bound, LeftAuthority75282.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75390.bound, LeftAuthority75282.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75390.actual selector witness, LeftAuthority75282.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75394

namespace LeftBound75398
def owner : Owner := ⟨.program ⟨214⟩, ⟨6811⟩⟩
def transferEvent : Nat := 75398
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75396 .coefficient, .predecessor 1 75397 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75396 .coefficient)
      LeftBound75394.bound (LeftBound75394.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75395RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75394.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75394.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75397 .coefficient)
      LeftAuthority75279.bound (LeftAuthority75279.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75279.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75279.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75394.bound, LeftAuthority75279.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75394.bound, LeftAuthority75279.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75394.actual selector witness, LeftAuthority75279.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75398

namespace LeftBound75402
def owner : Owner := ⟨.program ⟨214⟩, ⟨18646⟩⟩
def transferEvent : Nat := 75402
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75400 .coefficient, .predecessor 1 75401 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75400 .coefficient)
      LeftBound75398.bound (LeftBound75398.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75399RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75398.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75398.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75401 .coefficient)
      LeftBound75258.bound (LeftBound75258.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75277RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75258.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75258.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75398.bound, LeftBound75258.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75398.bound, LeftBound75258.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75398.actual selector witness, LeftBound75258.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75402

namespace LeftBound75406
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def transferEvent : Nat := 75406
def frameStart : Nat := 74728
def rule : BoundRule := .product (.predecessor 0 75404 .coefficient) (.predecessor 1 75405 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75404 .coefficient)
      LeftBound75402.bound (LeftBound75402.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75402.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75402.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75405 .coefficient)
      LeftAuthority75243.bound (LeftAuthority75243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75243.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75243.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound75402.bound LeftAuthority75243.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75402.bound, LeftAuthority75243.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound75402.actual selector witness) * (LeftAuthority75243.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound75406

namespace LeftBound75485
def owner : Owner := ⟨.program ⟨214⟩, ⟨18493⟩⟩
def transferEvent : Nat := 75485
def frameStart : Nat := 74728
def rule : BoundRule := .product (.predecessor 0 75483 .coefficient) (.predecessor 1 75484 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75483 .coefficient)
      LeftAuthority75254.bound (LeftAuthority75254.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events293.exact75255RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75254.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75254.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75484 .coefficient)
      LeftAuthority75481.bound (LeftAuthority75481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75481.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75481.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority75254.bound LeftAuthority75481.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority75254.bound, LeftAuthority75481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority75254.actual selector witness) * (LeftAuthority75481.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound75485

namespace LeftBound75493
def owner : Owner := ⟨.program ⟨214⟩, ⟨18494⟩⟩
def transferEvent : Nat := 75493
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75491 .coefficient, .predecessor 1 75492 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75491 .coefficient)
      LeftAuthority75489.bound (LeftAuthority75489.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75489.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75489.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75492 .coefficient)
      LeftBound75485.bound (LeftBound75485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75487RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75485.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75485.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority75489.bound, LeftBound75485.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority75489.bound, LeftBound75485.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority75489.actual selector witness, LeftBound75485.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75493

namespace LeftBound75497
def owner : Owner := ⟨.program ⟨214⟩, ⟨18680⟩⟩
def transferEvent : Nat := 75497
def frameStart : Nat := 74728
def rule : BoundRule := .sum [.predecessor 0 75495 .coefficient, .predecessor 1 75496 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75495 .coefficient)
      LeftBound75493.bound (LeftBound75493.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75493.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75496 .coefficient)
      LeftBound75406.bound (LeftBound75406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events294.exact75479RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75406.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75406.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75493.bound, LeftBound75406.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75493.bound, LeftBound75406.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75493.actual selector witness, LeftBound75406.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75497

namespace LeftBound75544
def owner : Owner := ⟨.program ⟨214⟩, ⟨30101⟩⟩
def transferEvent : Nat := 75544
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 75542 .coefficient, .predecessor 1 75543 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75542 .coefficient)
      LeftBound74135.bound (LeftBound74135.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75541RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74135.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75543 .coefficient)
      LeftBound74050.bound (LeftBound74050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events289.exact74125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74050.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74050.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74135.bound, LeftBound74050.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74135.bound, LeftBound74050.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound74135.actual selector witness, LeftBound74050.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75544

namespace LeftBound75581
def owner : Owner := ⟨.program ⟨214⟩, ⟨30101⟩⟩
def transferEvent : Nat := 75581
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 75541 .summary, .result 74125 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 75541 .summary)
      LeftBound74137.bound (LeftBound74137.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨18558⟩⟩) (rawTerms := some (Proof.Events295.exact75541RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound74137.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 74125 .summary)
      LeftBound74052.bound (LeftBound74052.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30100⟩⟩) (rawTerms := some (Proof.Events289.exact74125RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound74052.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74137.bound, LeftBound74052.bound]
def bound : CoeffClass := .finite ⟨85361036953731455419885957120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74137.bound, LeftBound74052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound74137.actual selector witness, LeftBound74052.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75581

namespace LeftBound75585
def owner : Owner := ⟨.program ⟨214⟩, ⟨30102⟩⟩
def transferEvent : Nat := 75585
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 75583 .coefficient) (.predecessor 1 75584 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75583 .coefficient)
      LeftBound75544.bound (LeftBound75544.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75582RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75544.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75544.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75584 .coefficient)
      LeftBound5498.bound (LeftBound5498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5499RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5498.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5498.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound75544.bound LeftBound5498.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75544.bound, LeftBound5498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound75544.actual selector witness) * (LeftBound5498.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound75585

namespace LeftBound75586
def owner : Owner := ⟨.program ⟨214⟩, ⟨30102⟩⟩
def transferEvent : Nat := 75586
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩ [⟨.result 5495 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5495 .coefficient)
      LeftAuthority5494.bound (LeftAuthority5494.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6651⟩⟩) (rawTerms := some (Proof.Events021.exact5495RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5494.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5494.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5494.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5494.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5494.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound75586

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
