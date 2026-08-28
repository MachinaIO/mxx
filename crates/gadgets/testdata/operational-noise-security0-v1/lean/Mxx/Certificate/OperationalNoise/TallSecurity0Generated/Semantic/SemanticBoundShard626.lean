import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard583
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard625

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound92330
def owner : Owner := ⟨.program ⟨214⟩, ⟨21331⟩⟩
def transferEvent : Nat := 92330
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 92329) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 92329)
      LeftBound92329.bound (LeftBound92329.actual selector witness) := by
  exact .transfer (LeftBound92329.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound92329.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound92329.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound92329.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92330

namespace LeftBound92425
def owner : Owner := ⟨.program ⟨214⟩, ⟨15941⟩⟩
def transferEvent : Nat := 92425
def frameStart : Nat := 92386
def rule : BoundRule := .identity (.predecessor 0 92424 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92424 .coefficient)
      LeftAuthority92422.bound (LeftAuthority92422.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92423RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92422.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92422.derived selector witness)

def rawBound : CoeffClass := LeftAuthority92422.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92422.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority92422.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound92425

namespace LeftBound92442
def owner : Owner := ⟨.program ⟨214⟩, ⟨16015⟩⟩
def transferEvent : Nat := 92442
def frameStart : Nat := 92386
def rule : BoundRule := .sum [.predecessor 0 92440 .coefficient, .predecessor 1 92441 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92440 .coefficient)
      LeftBound92425.bound (LeftBound92425.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound92425.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92441 .coefficient)
      LeftAuthority92438.bound (LeftAuthority92438.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority92438.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92425.bound, LeftAuthority92438.bound]
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92425.bound, LeftAuthority92438.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound92425.actual selector witness, LeftAuthority92438.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92442

namespace LeftBound92445
def owner : Owner := ⟨.program ⟨214⟩, ⟨16016⟩⟩
def transferEvent : Nat := 92445
def frameStart : Nat := 92386
def rule : BoundRule := .identity (.predecessor 0 92444 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92444 .coefficient)
      LeftBound92442.bound (LeftBound92442.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound92442.derived selector witness)

def rawBound : CoeffClass := LeftBound92442.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92442.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound92442.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound92445

namespace LeftBound92451
def owner : Owner := ⟨.program ⟨214⟩, ⟨16017⟩⟩
def transferEvent : Nat := 92451
def frameStart : Nat := 92386
def rule : BoundRule := .product (.predecessor 0 92449 .coefficient) (.predecessor 1 92450 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92449 .coefficient)
      LeftAuthority92447.bound (LeftAuthority92447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92448RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92447.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92447.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92450 .coefficient)
      LeftBound92445.bound (LeftBound92445.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92445.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92445.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority92447.bound LeftBound92445.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92447.bound, LeftBound92445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority92447.actual selector witness) * (LeftBound92445.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92451

namespace LeftBound92459
def owner : Owner := ⟨.program ⟨214⟩, ⟨16018⟩⟩
def transferEvent : Nat := 92459
def frameStart : Nat := 92386
def rule : BoundRule := .sum [.predecessor 0 92457 .coefficient, .predecessor 1 92458 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92457 .coefficient)
      LeftAuthority92455.bound (LeftAuthority92455.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92456RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92455.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92455.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92458 .coefficient)
      LeftBound92451.bound (LeftBound92451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92453RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92451.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92451.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority92455.bound, LeftBound92451.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92455.bound, LeftBound92451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority92455.actual selector witness, LeftBound92451.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92459

namespace LeftBound92463
def owner : Owner := ⟨.program ⟨214⟩, ⟨27860⟩⟩
def transferEvent : Nat := 92463
def frameStart : Nat := 92386
def rule : BoundRule := .product (.predecessor 0 92461 .coefficient) (.predecessor 1 92462 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92461 .coefficient)
      LeftBound92459.bound (LeftBound92459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92462 .coefficient)
      LeftAuthority92436.bound (LeftAuthority92436.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92437RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92436.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92436.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound92459.bound LeftAuthority92436.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92459.bound, LeftAuthority92436.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound92459.actual selector witness) * (LeftAuthority92436.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92463

namespace LeftBound92474
def owner : Owner := ⟨.program ⟨214⟩, ⟨17167⟩⟩
def transferEvent : Nat := 92474
def frameStart : Nat := 92386
def rule : BoundRule := .product (.predecessor 0 92472 .coefficient) (.predecessor 1 92473 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92472 .coefficient)
      LeftAuthority92447.bound (LeftAuthority92447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92448RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92447.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92447.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92473 .coefficient)
      LeftAuthority92470.bound (LeftAuthority92470.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92471RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92470.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92470.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority92447.bound LeftAuthority92470.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92447.bound, LeftAuthority92470.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority92447.actual selector witness) * (LeftAuthority92470.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92474

namespace LeftBound92482
def owner : Owner := ⟨.program ⟨214⟩, ⟨17168⟩⟩
def transferEvent : Nat := 92482
def frameStart : Nat := 92386
def rule : BoundRule := .sum [.predecessor 0 92480 .coefficient, .predecessor 1 92481 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92480 .coefficient)
      LeftAuthority92478.bound (LeftAuthority92478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92479RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92478.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92478.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92481 .coefficient)
      LeftBound92474.bound (LeftBound92474.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92474.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92474.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority92478.bound, LeftBound92474.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92478.bound, LeftBound92474.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority92478.actual selector witness, LeftBound92474.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92482

namespace LeftBound92486
def owner : Owner := ⟨.program ⟨214⟩, ⟨27865⟩⟩
def transferEvent : Nat := 92486
def frameStart : Nat := 92386
def rule : BoundRule := .sum [.predecessor 0 92484 .coefficient, .predecessor 1 92485 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92484 .coefficient)
      LeftBound92482.bound (LeftBound92482.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92482.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92482.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92485 .coefficient)
      LeftBound92463.bound (LeftBound92463.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92463.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92463.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92482.bound, LeftBound92463.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92482.bound, LeftBound92463.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound92482.actual selector witness, LeftBound92463.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92486

namespace LeftBound92499
def owner : Owner := ⟨.program ⟨214⟩, ⟨27862⟩⟩
def transferEvent : Nat := 92499
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 92497 .coefficient, .predecessor 1 92498 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92497 .coefficient)
      LeftBound92328.bound (LeftBound92328.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92496RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92328.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92328.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92498 .coefficient)
      LeftBound92311.bound (LeftBound92311.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events360.exact92318RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92311.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92311.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92328.bound, LeftBound92311.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92328.bound, LeftBound92311.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound92328.actual selector witness, LeftBound92311.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92499

namespace LeftBound92502
def owner : Owner := ⟨.program ⟨214⟩, ⟨27862⟩⟩
def transferEvent : Nat := 92502
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 92496 .summary, .result 92318 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92496 .summary)
      LeftBound92330.bound (LeftBound92330.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21331⟩⟩) (rawTerms := some (Proof.Events361.exact92496RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92330.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92318 .summary)
      LeftBound92313.bound (LeftBound92313.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27861⟩⟩) (rawTerms := some (Proof.Events360.exact92318RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92313.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92330.bound, LeftBound92313.bound]
def bound : CoeffClass := .finite ⟨1292068473939586330624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92330.bound, LeftBound92313.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound92330.actual selector witness, LeftBound92313.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92502

namespace LeftBound92506
def owner : Owner := ⟨.program ⟨214⟩, ⟨27863⟩⟩
def transferEvent : Nat := 92506
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 92504 .coefficient) (.predecessor 1 92505 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92504 .coefficient)
      LeftBound92499.bound (LeftBound92499.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92499.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92499.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92505 .coefficient)
      LeftBound5718.bound (LeftBound5718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5718.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5718.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound92499.bound LeftBound5718.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92499.bound, LeftBound5718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound92499.actual selector witness) * (LeftBound5718.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92506

namespace LeftBound92507
def owner : Owner := ⟨.program ⟨214⟩, ⟨27863⟩⟩
def transferEvent : Nat := 92507
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩ [⟨.result 5715 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5715 .coefficient)
      LeftAuthority5714.bound (LeftAuthority5714.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6641⟩⟩) (rawTerms := some (Proof.Events022.exact5715RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5714.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5714.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5714.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5714.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5714.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound92507

namespace LeftBound92508
def owner : Owner := ⟨.program ⟨214⟩, ⟨27863⟩⟩
def transferEvent : Nat := 92508
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 92503 .summary) (.transfer 92507) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92503 .summary)
      LeftBound92502.bound (LeftBound92502.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27862⟩⟩) (rawTerms := some (Proof.Events361.exact92503RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92502.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 92507)
      LeftBound92507.bound (LeftBound92507.actual selector witness) := by
  exact .transfer (LeftBound92507.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound92502.bound LeftBound92507.bound
def bound : CoeffClass := .finite ⟨4741911972453864866771369984, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92502.bound, LeftBound92507.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound92502.actual selector witness) * (LeftBound92507.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92508

namespace LeftBound92523
def owner : Owner := ⟨.program ⟨214⟩, ⟨27644⟩⟩
def transferEvent : Nat := 92523
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 92521 .coefficient) (.predecessor 1 92522 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92521 .coefficient)
      LeftBound85472.bound (LeftBound85472.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85472.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85472.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92522 .coefficient)
      LeftAuthority92519.bound (LeftAuthority92519.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92519.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92519.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound85472.bound LeftAuthority92519.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85472.bound, LeftAuthority92519.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound85472.actual selector witness) * (LeftAuthority92519.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92523

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
