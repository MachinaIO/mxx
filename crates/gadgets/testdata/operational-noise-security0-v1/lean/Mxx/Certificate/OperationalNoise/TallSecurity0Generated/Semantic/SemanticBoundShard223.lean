import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard185
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard222

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound34402
def owner : Owner := ⟨.program ⟨214⟩, ⟨15789⟩⟩
def transferEvent : Nat := 34402
def frameStart : Nat := 34346
def rule : BoundRule := .sum [.predecessor 0 34400 .coefficient, .predecessor 1 34401 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34400 .coefficient)
      LeftBound34385.bound (LeftBound34385.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound34385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34401 .coefficient)
      LeftAuthority34398.bound (LeftAuthority34398.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority34398.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound34385.bound, LeftAuthority34398.bound]
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34385.bound, LeftAuthority34398.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound34385.actual selector witness, LeftAuthority34398.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34402

namespace LeftBound34405
def owner : Owner := ⟨.program ⟨214⟩, ⟨15790⟩⟩
def transferEvent : Nat := 34405
def frameStart : Nat := 34346
def rule : BoundRule := .identity (.predecessor 0 34404 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34404 .coefficient)
      LeftBound34402.bound (LeftBound34402.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound34402.derived selector witness)

def rawBound : CoeffClass := LeftBound34402.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34402.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound34402.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound34405

namespace LeftBound34411
def owner : Owner := ⟨.program ⟨214⟩, ⟨15791⟩⟩
def transferEvent : Nat := 34411
def frameStart : Nat := 34346
def rule : BoundRule := .product (.predecessor 0 34409 .coefficient) (.predecessor 1 34410 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34409 .coefficient)
      LeftAuthority34407.bound (LeftAuthority34407.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34408RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34407.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34407.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34410 .coefficient)
      LeftBound34405.bound (LeftBound34405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34405.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34405.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority34407.bound LeftBound34405.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34407.bound, LeftBound34405.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority34407.actual selector witness) * (LeftBound34405.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34411

namespace LeftBound34419
def owner : Owner := ⟨.program ⟨214⟩, ⟨15792⟩⟩
def transferEvent : Nat := 34419
def frameStart : Nat := 34346
def rule : BoundRule := .sum [.predecessor 0 34417 .coefficient, .predecessor 1 34418 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34417 .coefficient)
      LeftAuthority34415.bound (LeftAuthority34415.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34416RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34415.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34415.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34418 .coefficient)
      LeftBound34411.bound (LeftBound34411.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34413RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34411.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34411.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority34415.bound, LeftBound34411.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34415.bound, LeftBound34411.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority34415.actual selector witness, LeftBound34411.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34419

namespace LeftBound34423
def owner : Owner := ⟨.program ⟨214⟩, ⟨27465⟩⟩
def transferEvent : Nat := 34423
def frameStart : Nat := 34346
def rule : BoundRule := .product (.predecessor 0 34421 .coefficient) (.predecessor 1 34422 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34421 .coefficient)
      LeftBound34419.bound (LeftBound34419.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34419.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34419.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34422 .coefficient)
      LeftAuthority34396.bound (LeftAuthority34396.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34397RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34396.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34396.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound34419.bound LeftAuthority34396.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34419.bound, LeftAuthority34396.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound34419.actual selector witness) * (LeftAuthority34396.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34423

namespace LeftBound34434
def owner : Owner := ⟨.program ⟨214⟩, ⟨17452⟩⟩
def transferEvent : Nat := 34434
def frameStart : Nat := 34346
def rule : BoundRule := .product (.predecessor 0 34432 .coefficient) (.predecessor 1 34433 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34432 .coefficient)
      LeftAuthority34407.bound (LeftAuthority34407.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34408RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34407.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34407.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34433 .coefficient)
      LeftAuthority34430.bound (LeftAuthority34430.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34431RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34430.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34430.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority34407.bound LeftAuthority34430.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34407.bound, LeftAuthority34430.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority34407.actual selector witness) * (LeftAuthority34430.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34434

namespace LeftBound34442
def owner : Owner := ⟨.program ⟨214⟩, ⟨17453⟩⟩
def transferEvent : Nat := 34442
def frameStart : Nat := 34346
def rule : BoundRule := .sum [.predecessor 0 34440 .coefficient, .predecessor 1 34441 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34440 .coefficient)
      LeftAuthority34438.bound (LeftAuthority34438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34439RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34438.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34438.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34441 .coefficient)
      LeftBound34434.bound (LeftBound34434.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34436RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34434.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34434.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority34438.bound, LeftBound34434.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34438.bound, LeftBound34434.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority34438.actual selector witness, LeftBound34434.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34442

namespace LeftBound34446
def owner : Owner := ⟨.program ⟨214⟩, ⟨27470⟩⟩
def transferEvent : Nat := 34446
def frameStart : Nat := 34346
def rule : BoundRule := .sum [.predecessor 0 34444 .coefficient, .predecessor 1 34445 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34444 .coefficient)
      LeftBound34442.bound (LeftBound34442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34443RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34442.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34442.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34445 .coefficient)
      LeftBound34423.bound (LeftBound34423.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34428RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34423.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34423.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound34442.bound, LeftBound34423.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34442.bound, LeftBound34423.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound34442.actual selector witness, LeftBound34423.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34446

namespace LeftBound34459
def owner : Owner := ⟨.program ⟨214⟩, ⟨27467⟩⟩
def transferEvent : Nat := 34459
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 34457 .coefficient, .predecessor 1 34458 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34457 .coefficient)
      LeftBound34288.bound (LeftBound34288.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34456RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34288.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34288.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34458 .coefficient)
      LeftBound34271.bound (LeftBound34271.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events133.exact34278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34271.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34271.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound34288.bound, LeftBound34271.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34288.bound, LeftBound34271.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound34288.actual selector witness, LeftBound34271.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34459

namespace LeftBound34462
def owner : Owner := ⟨.program ⟨214⟩, ⟨27467⟩⟩
def transferEvent : Nat := 34462
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 34456 .summary, .result 34278 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 34456 .summary)
      LeftBound34290.bound (LeftBound34290.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21055⟩⟩) (rawTerms := some (Proof.Events134.exact34456RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34290.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 34278 .summary)
      LeftBound34273.bound (LeftBound34273.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27466⟩⟩) (rawTerms := some (Proof.Events133.exact34278RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34273.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound34290.bound, LeftBound34273.bound]
def bound : CoeffClass := .finite ⟨1292001236604524572672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34290.bound, LeftBound34273.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound34290.actual selector witness, LeftBound34273.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34462

namespace LeftBound34466
def owner : Owner := ⟨.program ⟨214⟩, ⟨27468⟩⟩
def transferEvent : Nat := 34466
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 34464 .coefficient) (.predecessor 1 34465 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34464 .coefficient)
      LeftBound34459.bound (LeftBound34459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34465 .coefficient)
      LeftBound5758.bound (LeftBound5758.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5759RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5758.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5758.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound34459.bound LeftBound5758.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34459.bound, LeftBound5758.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound34459.actual selector witness) * (LeftBound5758.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34466

namespace LeftBound34467
def owner : Owner := ⟨.program ⟨214⟩, ⟨27468⟩⟩
def transferEvent : Nat := 34467
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩ [⟨.result 5755 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5755 .coefficient)
      LeftAuthority5754.bound (LeftAuthority5754.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6647⟩⟩) (rawTerms := some (Proof.Events022.exact5755RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5754.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5754.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5754.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5754.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5754.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound34467

namespace LeftBound34468
def owner : Owner := ⟨.program ⟨214⟩, ⟨27468⟩⟩
def transferEvent : Nat := 34468
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 34463 .summary) (.transfer 34467) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 34463 .summary)
      LeftBound34462.bound (LeftBound34462.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27467⟩⟩) (rawTerms := some (Proof.Events134.exact34463RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34462.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 34467)
      LeftBound34467.bound (LeftBound34467.actual selector witness) := by
  exact .transfer (LeftBound34467.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound34462.bound LeftBound34467.bound
def bound : CoeffClass := .finite ⟨4741665210358390854099402752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34462.bound, LeftBound34467.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound34462.actual selector witness) * (LeftBound34467.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34468

namespace LeftBound34483
def owner : Owner := ⟨.program ⟨214⟩, ⟨27249⟩⟩
def transferEvent : Nat := 34483
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 34481 .coefficient) (.predecessor 1 34482 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34481 .coefficient)
      LeftBound27960.bound (LeftBound27960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact27964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27960.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34482 .coefficient)
      LeftAuthority34479.bound (LeftAuthority34479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34479.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34479.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound27960.bound LeftAuthority34479.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27960.bound, LeftAuthority34479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound27960.actual selector witness) * (LeftAuthority34479.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34483

namespace LeftBound34484
def owner : Owner := ⟨.program ⟨214⟩, ⟨27249⟩⟩
def transferEvent : Nat := 34484
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27247⟩⟩]⟩ [⟨.result 34480 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 34480 .coefficient)
      LeftAuthority34479.bound (LeftAuthority34479.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27247⟩⟩) (rawTerms := some (Proof.Events134.exact34480RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34479.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34479.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority34479.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority34479.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound34484

namespace LeftBound34485
def owner : Owner := ⟨.program ⟨214⟩, ⟨27249⟩⟩
def transferEvent : Nat := 34485
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 27964 .summary) (.transfer 34484) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27964 .summary)
      LeftBound27963.bound (LeftBound27963.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25852⟩⟩) (rawTerms := some (Proof.Events109.exact27964RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27963.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 34484)
      LeftBound34484.bound (LeftBound34484.actual selector witness) := by
  exact .transfer (LeftBound34484.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound27963.bound LeftBound34484.bound
def bound : CoeffClass := .finite ⟨1291978822348200476672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27963.bound, LeftBound34484.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound27963.actual selector witness) * (LeftBound34484.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34485

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
