import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard294
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard326

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound49434
def owner : Owner := ⟨.program ⟨214⟩, ⟨15431⟩⟩
def transferEvent : Nat := 49434
def frameStart : Nat := 49395
def rule : BoundRule := .identity (.predecessor 0 49433 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49433 .coefficient)
      LeftAuthority49431.bound (LeftAuthority49431.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49432RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49431.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49431.derived selector witness)

def rawBound : CoeffClass := LeftAuthority49431.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49431.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority49431.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound49434

namespace LeftBound49451
def owner : Owner := ⟨.program ⟨214⟩, ⟨15470⟩⟩
def transferEvent : Nat := 49451
def frameStart : Nat := 49395
def rule : BoundRule := .sum [.predecessor 0 49449 .coefficient, .predecessor 1 49450 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49449 .coefficient)
      LeftBound49434.bound (LeftBound49434.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound49434.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49450 .coefficient)
      LeftAuthority49447.bound (LeftAuthority49447.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority49447.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound49434.bound, LeftAuthority49447.bound]
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49434.bound, LeftAuthority49447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound49434.actual selector witness, LeftAuthority49447.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49451

namespace LeftBound49454
def owner : Owner := ⟨.program ⟨214⟩, ⟨15471⟩⟩
def transferEvent : Nat := 49454
def frameStart : Nat := 49395
def rule : BoundRule := .identity (.predecessor 0 49453 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49453 .coefficient)
      LeftBound49451.bound (LeftBound49451.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound49451.derived selector witness)

def rawBound : CoeffClass := LeftBound49451.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound49451.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound49454

namespace LeftBound49460
def owner : Owner := ⟨.program ⟨214⟩, ⟨15472⟩⟩
def transferEvent : Nat := 49460
def frameStart : Nat := 49395
def rule : BoundRule := .product (.predecessor 0 49458 .coefficient) (.predecessor 1 49459 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49458 .coefficient)
      LeftAuthority49456.bound (LeftAuthority49456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49456.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49456.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49459 .coefficient)
      LeftBound49454.bound (LeftBound49454.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49455RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49454.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49454.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority49456.bound LeftBound49454.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49456.bound, LeftBound49454.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority49456.actual selector witness) * (LeftBound49454.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49460

namespace LeftBound49468
def owner : Owner := ⟨.program ⟨214⟩, ⟨15473⟩⟩
def transferEvent : Nat := 49468
def frameStart : Nat := 49395
def rule : BoundRule := .sum [.predecessor 0 49466 .coefficient, .predecessor 1 49467 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49466 .coefficient)
      LeftAuthority49464.bound (LeftAuthority49464.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49465RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49464.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49464.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49467 .coefficient)
      LeftBound49460.bound (LeftBound49460.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49460.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49460.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority49464.bound, LeftBound49460.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49464.bound, LeftBound49460.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority49464.actual selector witness, LeftBound49460.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49468

namespace LeftBound49472
def owner : Owner := ⟨.program ⟨214⟩, ⟨27018⟩⟩
def transferEvent : Nat := 49472
def frameStart : Nat := 49395
def rule : BoundRule := .product (.predecessor 0 49470 .coefficient) (.predecessor 1 49471 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49470 .coefficient)
      LeftBound49468.bound (LeftBound49468.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49469RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49468.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49468.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49471 .coefficient)
      LeftAuthority49445.bound (LeftAuthority49445.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49445.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49445.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound49468.bound LeftAuthority49445.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49468.bound, LeftAuthority49445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound49468.actual selector witness) * (LeftAuthority49445.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49472

namespace LeftBound49483
def owner : Owner := ⟨.program ⟨214⟩, ⟨15529⟩⟩
def transferEvent : Nat := 49483
def frameStart : Nat := 49395
def rule : BoundRule := .product (.predecessor 0 49481 .coefficient) (.predecessor 1 49482 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49481 .coefficient)
      LeftAuthority49456.bound (LeftAuthority49456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49456.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49456.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49482 .coefficient)
      LeftAuthority49479.bound (LeftAuthority49479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49479.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49479.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority49456.bound LeftAuthority49479.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49456.bound, LeftAuthority49479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority49456.actual selector witness) * (LeftAuthority49479.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49483

namespace LeftBound49491
def owner : Owner := ⟨.program ⟨214⟩, ⟨15530⟩⟩
def transferEvent : Nat := 49491
def frameStart : Nat := 49395
def rule : BoundRule := .sum [.predecessor 0 49489 .coefficient, .predecessor 1 49490 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49489 .coefficient)
      LeftAuthority49487.bound (LeftAuthority49487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49487.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49487.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49490 .coefficient)
      LeftBound49483.bound (LeftBound49483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49485RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49483.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49483.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority49487.bound, LeftBound49483.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49487.bound, LeftBound49483.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority49487.actual selector witness, LeftBound49483.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49491

namespace LeftBound49495
def owner : Owner := ⟨.program ⟨214⟩, ⟨27023⟩⟩
def transferEvent : Nat := 49495
def frameStart : Nat := 49395
def rule : BoundRule := .sum [.predecessor 0 49493 .coefficient, .predecessor 1 49494 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49493 .coefficient)
      LeftBound49491.bound (LeftBound49491.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49492RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49491.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49491.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49494 .coefficient)
      LeftBound49472.bound (LeftBound49472.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49477RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49472.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49472.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound49491.bound, LeftBound49472.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49491.bound, LeftBound49472.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound49491.actual selector witness, LeftBound49472.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49495

namespace LeftBound49508
def owner : Owner := ⟨.program ⟨214⟩, ⟨27020⟩⟩
def transferEvent : Nat := 49508
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 49506 .coefficient, .predecessor 1 49507 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49506 .coefficient)
      LeftBound49337.bound (LeftBound49337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49505RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49337.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49507 .coefficient)
      LeftBound49320.bound (LeftBound49320.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events192.exact49327RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49320.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49320.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound49337.bound, LeftBound49320.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49337.bound, LeftBound49320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound49337.actual selector witness, LeftBound49320.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49508

namespace LeftBound49511
def owner : Owner := ⟨.program ⟨214⟩, ⟨27020⟩⟩
def transferEvent : Nat := 49511
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 49505 .summary, .result 49327 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49505 .summary)
      LeftBound49339.bound (LeftBound49339.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20763⟩⟩) (rawTerms := some (Proof.Events193.exact49505RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound49339.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49327 .summary)
      LeftBound49322.bound (LeftBound49322.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27019⟩⟩) (rawTerms := some (Proof.Events192.exact49327RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound49322.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound49339.bound, LeftBound49322.bound]
def bound : CoeffClass := .finite ⟨1291933999269462814720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49339.bound, LeftBound49322.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound49339.actual selector witness, LeftBound49322.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49511

namespace LeftBound49515
def owner : Owner := ⟨.program ⟨214⟩, ⟨27021⟩⟩
def transferEvent : Nat := 49515
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 49513 .coefficient) (.predecessor 1 49514 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49513 .coefficient)
      LeftBound49508.bound (LeftBound49508.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49508.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49514 .coefficient)
      LeftBound5798.bound (LeftBound5798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5798.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5798.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound49508.bound LeftBound5798.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49508.bound, LeftBound5798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound49508.actual selector witness) * (LeftBound5798.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49515

namespace LeftBound49516
def owner : Owner := ⟨.program ⟨214⟩, ⟨27021⟩⟩
def transferEvent : Nat := 49516
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩ [⟨.result 5795 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5795 .coefficient)
      LeftAuthority5794.bound (LeftAuthority5794.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6655⟩⟩) (rawTerms := some (Proof.Events022.exact5795RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5794.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5794.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5794.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5794.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5794.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound49516

namespace LeftBound49517
def owner : Owner := ⟨.program ⟨214⟩, ⟨27021⟩⟩
def transferEvent : Nat := 49517
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 49512 .summary) (.transfer 49516) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49512 .summary)
      LeftBound49511.bound (LeftBound49511.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27020⟩⟩) (rawTerms := some (Proof.Events193.exact49512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound49511.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 49516)
      LeftBound49516.bound (LeftBound49516.actual selector witness) := by
  exact .transfer (LeftBound49516.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound49511.bound LeftBound49516.bound
def bound : CoeffClass := .finite ⟨4741418448262916841427435520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49511.bound, LeftBound49516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound49511.actual selector witness) * (LeftBound49516.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49517

namespace LeftBound49532
def owner : Owner := ⟨.program ⟨214⟩, ⟨26802⟩⟩
def transferEvent : Nat := 49532
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 49530 .coefficient) (.predecessor 1 49531 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49530 .coefficient)
      LeftBound43549.bound (LeftBound43549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43549.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43549.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49531 .coefficient)
      LeftAuthority49528.bound (LeftAuthority49528.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49529RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49528.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49528.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43549.bound LeftAuthority49528.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43549.bound, LeftAuthority49528.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43549.actual selector witness) * (LeftAuthority49528.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49532

namespace LeftBound49533
def owner : Owner := ⟨.program ⟨214⟩, ⟨26802⟩⟩
def transferEvent : Nat := 49533
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26800⟩⟩]⟩ [⟨.result 49529 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49529 .coefficient)
      LeftAuthority49528.bound (LeftAuthority49528.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26800⟩⟩) (rawTerms := some (Proof.Events193.exact49529RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49528.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49528.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority49528.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49528.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority49528.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound49533

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
