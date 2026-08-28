import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard039
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard106

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound17425
def owner : Owner := ⟨.program ⟨214⟩, ⟨16888⟩⟩
def transferEvent : Nat := 17425
def frameStart : Nat := 17386
def rule : BoundRule := .identity (.predecessor 0 17424 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17424 .coefficient)
      LeftAuthority17422.bound (LeftAuthority17422.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17423RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17422.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17422.derived selector witness)

def rawBound : CoeffClass := LeftAuthority17422.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17422.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority17422.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound17425

namespace LeftBound17442
def owner : Owner := ⟨.program ⟨214⟩, ⟨16983⟩⟩
def transferEvent : Nat := 17442
def frameStart : Nat := 17386
def rule : BoundRule := .sum [.predecessor 0 17440 .coefficient, .predecessor 1 17441 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17440 .coefficient)
      LeftBound17425.bound (LeftBound17425.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound17425.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17441 .coefficient)
      LeftAuthority17438.bound (LeftAuthority17438.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority17438.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound17425.bound, LeftAuthority17438.bound]
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17425.bound, LeftAuthority17438.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound17425.actual selector witness, LeftAuthority17438.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17442

namespace LeftBound17445
def owner : Owner := ⟨.program ⟨214⟩, ⟨16984⟩⟩
def transferEvent : Nat := 17445
def frameStart : Nat := 17386
def rule : BoundRule := .identity (.predecessor 0 17444 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17444 .coefficient)
      LeftBound17442.bound (LeftBound17442.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound17442.derived selector witness)

def rawBound : CoeffClass := LeftBound17442.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17442.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound17442.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound17445

namespace LeftBound17451
def owner : Owner := ⟨.program ⟨214⟩, ⟨16985⟩⟩
def transferEvent : Nat := 17451
def frameStart : Nat := 17386
def rule : BoundRule := .product (.predecessor 0 17449 .coefficient) (.predecessor 1 17450 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17449 .coefficient)
      LeftAuthority17447.bound (LeftAuthority17447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17448RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17447.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17447.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17450 .coefficient)
      LeftBound17445.bound (LeftBound17445.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17445.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17445.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority17447.bound LeftBound17445.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17447.bound, LeftBound17445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority17447.actual selector witness) * (LeftBound17445.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17451

namespace LeftBound17459
def owner : Owner := ⟨.program ⟨214⟩, ⟨16986⟩⟩
def transferEvent : Nat := 17459
def frameStart : Nat := 17386
def rule : BoundRule := .sum [.predecessor 0 17457 .coefficient, .predecessor 1 17458 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17457 .coefficient)
      LeftAuthority17455.bound (LeftAuthority17455.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17456RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17455.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17455.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17458 .coefficient)
      LeftBound17451.bound (LeftBound17451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17453RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17451.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17451.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority17455.bound, LeftBound17451.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17455.bound, LeftBound17451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority17455.actual selector witness, LeftBound17451.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17459

namespace LeftBound17463
def owner : Owner := ⟨.program ⟨214⟩, ⟨29865⟩⟩
def transferEvent : Nat := 17463
def frameStart : Nat := 17386
def rule : BoundRule := .product (.predecessor 0 17461 .coefficient) (.predecessor 1 17462 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17461 .coefficient)
      LeftBound17459.bound (LeftBound17459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17462 .coefficient)
      LeftAuthority17436.bound (LeftAuthority17436.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17437RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17436.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17436.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound17459.bound LeftAuthority17436.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17459.bound, LeftAuthority17436.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound17459.actual selector witness) * (LeftAuthority17436.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17463

namespace LeftBound17474
def owner : Owner := ⟨.program ⟨214⟩, ⟨16945⟩⟩
def transferEvent : Nat := 17474
def frameStart : Nat := 17386
def rule : BoundRule := .product (.predecessor 0 17472 .coefficient) (.predecessor 1 17473 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17472 .coefficient)
      LeftAuthority17447.bound (LeftAuthority17447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17448RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17447.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17447.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17473 .coefficient)
      LeftAuthority17470.bound (LeftAuthority17470.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17471RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17470.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17470.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority17447.bound LeftAuthority17470.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17447.bound, LeftAuthority17470.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority17447.actual selector witness) * (LeftAuthority17470.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17474

namespace LeftBound17482
def owner : Owner := ⟨.program ⟨214⟩, ⟨16946⟩⟩
def transferEvent : Nat := 17482
def frameStart : Nat := 17386
def rule : BoundRule := .sum [.predecessor 0 17480 .coefficient, .predecessor 1 17481 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17480 .coefficient)
      LeftAuthority17478.bound (LeftAuthority17478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17479RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17478.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17478.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17481 .coefficient)
      LeftBound17474.bound (LeftBound17474.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17474.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17474.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority17478.bound, LeftBound17474.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17478.bound, LeftBound17474.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority17478.actual selector witness, LeftBound17474.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17482

namespace LeftBound17486
def owner : Owner := ⟨.program ⟨214⟩, ⟨29870⟩⟩
def transferEvent : Nat := 17486
def frameStart : Nat := 17386
def rule : BoundRule := .sum [.predecessor 0 17484 .coefficient, .predecessor 1 17485 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17484 .coefficient)
      LeftBound17482.bound (LeftBound17482.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17482.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17482.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17485 .coefficient)
      LeftBound17463.bound (LeftBound17463.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17463.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17463.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound17482.bound, LeftBound17463.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17482.bound, LeftBound17463.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound17482.actual selector witness, LeftBound17463.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17486

namespace LeftBound17499
def owner : Owner := ⟨.program ⟨214⟩, ⟨29867⟩⟩
def transferEvent : Nat := 17499
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 17497 .coefficient, .predecessor 1 17498 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17497 .coefficient)
      LeftBound17328.bound (LeftBound17328.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17496RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17328.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17328.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17498 .coefficient)
      LeftBound17311.bound (LeftBound17311.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17318RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17311.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17311.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound17328.bound, LeftBound17311.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17328.bound, LeftBound17311.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound17328.actual selector witness, LeftBound17311.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17499

namespace LeftBound17502
def owner : Owner := ⟨.program ⟨214⟩, ⟨29867⟩⟩
def transferEvent : Nat := 17502
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 17496 .summary, .result 17318 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17496 .summary)
      LeftBound17330.bound (LeftBound17330.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22643⟩⟩) (rawTerms := some (Proof.Events068.exact17496RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17330.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17318 .summary)
      LeftBound17313.bound (LeftBound17313.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29866⟩⟩) (rawTerms := some (Proof.Events067.exact17318RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17313.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound17330.bound, LeftBound17313.bound]
def bound : CoeffClass := .finite ⟨1292516722839998050304, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17330.bound, LeftBound17313.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound17330.actual selector witness, LeftBound17313.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17502

namespace LeftBound17506
def owner : Owner := ⟨.program ⟨214⟩, ⟨29868⟩⟩
def transferEvent : Nat := 17506
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 17504 .coefficient) (.predecessor 1 17505 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17504 .coefficient)
      LeftBound17499.bound (LeftBound17499.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17499.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17499.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17505 .coefficient)
      LeftBound5538.bound (LeftBound5538.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5539RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5538.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5538.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound17499.bound LeftBound5538.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17499.bound, LeftBound5538.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound17499.actual selector witness) * (LeftBound5538.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17506

namespace LeftBound17507
def owner : Owner := ⟨.program ⟨214⟩, ⟨29868⟩⟩
def transferEvent : Nat := 17507
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6659⟩⟩]⟩ [⟨.result 5535 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5535 .coefficient)
      LeftAuthority5534.bound (LeftAuthority5534.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6659⟩⟩) (rawTerms := some (Proof.Events021.exact5535RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5534.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5534.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5534.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5534.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound17507

namespace LeftBound17508
def owner : Owner := ⟨.program ⟨214⟩, ⟨29868⟩⟩
def transferEvent : Nat := 17508
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 17503 .summary) (.transfer 17507) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17503 .summary)
      LeftBound17502.bound (LeftBound17502.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29867⟩⟩) (rawTerms := some (Proof.Events068.exact17503RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17502.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 17507)
      LeftBound17507.bound (LeftBound17507.actual selector witness) := by
  exact .transfer (LeftBound17507.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound17502.bound LeftBound17507.bound
def bound : CoeffClass := .finite ⟨4743557053090358284584484864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17502.bound, LeftBound17507.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound17502.actual selector witness) * (LeftBound17507.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17508

namespace LeftBound17523
def owner : Owner := ⟨.program ⟨214⟩, ⟨29649⟩⟩
def transferEvent : Nat := 17523
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 17521 .coefficient) (.predecessor 1 17522 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17521 .coefficient)
      LeftBound7745.bound (LeftBound7745.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events030.exact7749RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7745.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7745.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17522 .coefficient)
      LeftAuthority17519.bound (LeftAuthority17519.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17519.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17519.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound7745.bound LeftAuthority17519.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7745.bound, LeftAuthority17519.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound7745.actual selector witness) * (LeftAuthority17519.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17523

namespace LeftBound17524
def owner : Owner := ⟨.program ⟨214⟩, ⟨29649⟩⟩
def transferEvent : Nat := 17524
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29647⟩⟩]⟩ [⟨.result 17520 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17520 .coefficient)
      LeftAuthority17519.bound (LeftAuthority17519.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29647⟩⟩) (rawTerms := some (Proof.Events068.exact17520RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17519.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17519.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority17519.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17519.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority17519.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound17524

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
