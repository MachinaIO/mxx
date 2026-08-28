import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard076
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard682

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound99491
def owner : Owner := ⟨.program ⟨214⟩, ⟨15812⟩⟩
def transferEvent : Nat := 99491
def frameStart : Nat := 99464
def rule : BoundRule := .identity (.predecessor 0 99490 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99490 .coefficient)
      LeftAuthority99488.bound (LeftAuthority99488.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99489RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99488.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99488.derived selector witness)

def rawBound : CoeffClass := LeftAuthority99488.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99488.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority99488.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound99491

namespace LeftBound99508
def owner : Owner := ⟨.program ⟨214⟩, ⟨15888⟩⟩
def transferEvent : Nat := 99508
def frameStart : Nat := 99464
def rule : BoundRule := .sum [.predecessor 0 99506 .coefficient, .predecessor 1 99507 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99506 .coefficient)
      LeftBound99491.bound (LeftBound99491.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound99491.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99507 .coefficient)
      LeftAuthority99504.bound (LeftAuthority99504.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority99504.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99491.bound, LeftAuthority99504.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99491.bound, LeftAuthority99504.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99491.actual selector witness, LeftAuthority99504.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99508

namespace LeftBound99511
def owner : Owner := ⟨.program ⟨214⟩, ⟨15889⟩⟩
def transferEvent : Nat := 99511
def frameStart : Nat := 99464
def rule : BoundRule := .identity (.predecessor 0 99510 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99510 .coefficient)
      LeftBound99508.bound (LeftBound99508.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound99508.derived selector witness)

def rawBound : CoeffClass := LeftBound99508.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99508.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound99508.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound99511

namespace LeftBound99517
def owner : Owner := ⟨.program ⟨214⟩, ⟨15890⟩⟩
def transferEvent : Nat := 99517
def frameStart : Nat := 99464
def rule : BoundRule := .product (.predecessor 0 99515 .coefficient) (.predecessor 1 99516 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99515 .coefficient)
      LeftAuthority99513.bound (LeftAuthority99513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99514RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99513.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99513.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99516 .coefficient)
      LeftBound99511.bound (LeftBound99511.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99511.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99511.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority99513.bound LeftBound99511.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99513.bound, LeftBound99511.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority99513.actual selector witness) * (LeftBound99511.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99517

namespace LeftBound99525
def owner : Owner := ⟨.program ⟨214⟩, ⟨15891⟩⟩
def transferEvent : Nat := 99525
def frameStart : Nat := 99464
def rule : BoundRule := .sum [.predecessor 0 99523 .coefficient, .predecessor 1 99524 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99523 .coefficient)
      LeftAuthority99521.bound (LeftAuthority99521.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99522RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99521.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99521.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99524 .coefficient)
      LeftBound99517.bound (LeftBound99517.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99519RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99517.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99517.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority99521.bound, LeftBound99517.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99521.bound, LeftBound99517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority99521.actual selector witness, LeftBound99517.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99525

namespace LeftBound99529
def owner : Owner := ⟨.program ⟨214⟩, ⟨27615⟩⟩
def transferEvent : Nat := 99529
def frameStart : Nat := 99464
def rule : BoundRule := .product (.predecessor 0 99527 .coefficient) (.predecessor 1 99528 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99527 .coefficient)
      LeftBound99525.bound (LeftBound99525.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99526RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99525.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99525.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99528 .coefficient)
      LeftAuthority99502.bound (LeftAuthority99502.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99502.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99502.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99525.bound LeftAuthority99502.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99525.bound, LeftAuthority99502.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99525.actual selector witness) * (LeftAuthority99502.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99529

namespace LeftBound99540
def owner : Owner := ⟨.program ⟨214⟩, ⟨15861⟩⟩
def transferEvent : Nat := 99540
def frameStart : Nat := 99464
def rule : BoundRule := .product (.predecessor 0 99538 .coefficient) (.predecessor 1 99539 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99538 .coefficient)
      LeftAuthority99513.bound (LeftAuthority99513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99514RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99513.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99513.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99539 .coefficient)
      LeftAuthority99536.bound (LeftAuthority99536.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99537RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99536.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99536.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority99513.bound LeftAuthority99536.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99513.bound, LeftAuthority99536.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority99513.actual selector witness) * (LeftAuthority99536.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99540

namespace LeftBound99548
def owner : Owner := ⟨.program ⟨214⟩, ⟨15862⟩⟩
def transferEvent : Nat := 99548
def frameStart : Nat := 99464
def rule : BoundRule := .sum [.predecessor 0 99546 .coefficient, .predecessor 1 99547 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99546 .coefficient)
      LeftAuthority99544.bound (LeftAuthority99544.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99545RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99544.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99544.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99547 .coefficient)
      LeftBound99540.bound (LeftBound99540.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99542RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99540.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99540.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority99544.bound, LeftBound99540.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99544.bound, LeftBound99540.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority99544.actual selector witness, LeftBound99540.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99548

namespace LeftBound99552
def owner : Owner := ⟨.program ⟨214⟩, ⟨27619⟩⟩
def transferEvent : Nat := 99552
def frameStart : Nat := 99464
def rule : BoundRule := .sum [.predecessor 0 99550 .coefficient, .predecessor 1 99551 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99550 .coefficient)
      LeftBound99548.bound (LeftBound99548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99549RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99548.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99548.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99551 .coefficient)
      LeftBound99529.bound (LeftBound99529.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99534RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99529.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99529.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99548.bound, LeftBound99529.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99548.bound, LeftBound99529.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99548.actual selector witness, LeftBound99529.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99552

namespace LeftBound99565
def owner : Owner := ⟨.program ⟨214⟩, ⟨27617⟩⟩
def transferEvent : Nat := 99565
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99563 .coefficient, .predecessor 1 99564 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99563 .coefficient)
      LeftBound99418.bound (LeftBound99418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99562RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99418.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99564 .coefficient)
      LeftBound99401.bound (LeftBound99401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99408RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99401.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99401.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99418.bound, LeftBound99401.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99418.bound, LeftBound99401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99418.actual selector witness, LeftBound99401.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99565

namespace LeftBound99568
def owner : Owner := ⟨.program ⟨214⟩, ⟨27617⟩⟩
def transferEvent : Nat := 99568
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99562 .summary, .result 99408 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99562 .summary)
      LeftBound99420.bound (LeftBound99420.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21248⟩⟩) (rawTerms := some (Proof.Events388.exact99562RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99420.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99408 .summary)
      LeftBound99403.bound (LeftBound99403.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27616⟩⟩) (rawTerms := some (Proof.Events388.exact99408RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99403.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99420.bound, LeftBound99403.bound]
def bound : CoeffClass := .finite ⟨1292046061494565744640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99420.bound, LeftBound99403.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99420.actual selector witness, LeftBound99403.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99568

namespace LeftBound99592
def owner : Owner := ⟨.program ⟨214⟩, ⟨11290⟩⟩
def transferEvent : Nat := 99592
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 99590 .coefficient) (.predecessor 1 99591 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99590 .coefficient)
      LeftAuthority4841.bound (LeftAuthority4841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4841.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4841.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99591 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4841.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4841.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4841.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound99592

namespace LeftBound99597
def owner : Owner := ⟨.program ⟨214⟩, ⟨7114⟩⟩
def transferEvent : Nat := 99597
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 99595 .coefficient) (.predecessor 1 99596 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99595 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99596 .coefficient)
      LeftBound12483.bound (LeftBound12483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12483.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12483.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound12483.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound12483.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound12483.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99597

namespace LeftBound99602
def owner : Owner := ⟨.program ⟨214⟩, ⟨11291⟩⟩
def transferEvent : Nat := 99602
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99600 .coefficient, .predecessor 1 99601 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99600 .coefficient)
      LeftBound99597.bound (LeftBound99597.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99599RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99597.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99597.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99601 .coefficient)
      LeftBound99592.bound (LeftBound99592.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99594RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99592.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99592.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99597.bound, LeftBound99592.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99597.bound, LeftBound99592.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99597.actual selector witness, LeftBound99592.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99602

namespace LeftBound99606
def owner : Owner := ⟨.program ⟨214⟩, ⟨11292⟩⟩
def transferEvent : Nat := 99606
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99604 .coefficient, .predecessor 1 99605 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99604 .coefficient)
      LeftBound99602.bound (LeftBound99602.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99603RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99602.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99602.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99605 .coefficient)
      LeftBound12475.bound (LeftBound12475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12475.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99602.bound, LeftBound12475.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99602.bound, LeftBound12475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99602.actual selector witness, LeftBound12475.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99606

namespace LeftBound99607
def owner : Owner := ⟨.program ⟨214⟩, ⟨11292⟩⟩
def transferEvent : Nat := 99607
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨91⟩⟩]⟩ [⟨.result 12476 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12476 .coefficient)
      LeftBound12475.bound (LeftBound12475.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨91⟩⟩) (rawTerms := some (Proof.Events048.exact12476RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12475.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound12475.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound12475.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound99607

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
