import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard300

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound44468
def owner : Owner := ⟨.program ⟨214⟩, ⟨7833⟩⟩
def transferEvent : Nat := 44468
def frameStart : Nat := 44386
def rule : BoundRule := .product (.predecessor 0 44466 .coefficient) (.predecessor 1 44467 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44466 .coefficient)
      LeftBound44464.bound (LeftBound44464.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44465RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44464.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44464.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44467 .coefficient)
      LeftBound44461.bound (LeftBound44461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44461.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44461.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound44464.bound LeftBound44461.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44464.bound, LeftBound44461.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound44464.actual selector witness) * (LeftBound44461.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44468

namespace LeftBound44473
def owner : Owner := ⟨.program ⟨214⟩, ⟨10587⟩⟩
def transferEvent : Nat := 44473
def frameStart : Nat := 44386
def rule : BoundRule := .sum [.predecessor 0 44471 .coefficient, .predecessor 1 44472 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44471 .coefficient)
      LeftBound44468.bound (LeftBound44468.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44470RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44468.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44468.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44472 .coefficient)
      LeftBound44445.bound (LeftBound44445.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44447RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44445.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44445.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44468.bound, LeftBound44445.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44468.bound, LeftBound44445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44468.actual selector witness, LeftBound44445.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44473

namespace LeftBound44477
def owner : Owner := ⟨.program ⟨214⟩, ⟨24924⟩⟩
def transferEvent : Nat := 44477
def frameStart : Nat := 44386
def rule : BoundRule := .product (.predecessor 0 44475 .coefficient) (.predecessor 1 44476 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44475 .coefficient)
      LeftBound44473.bound (LeftBound44473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44473.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44476 .coefficient)
      LeftAuthority44430.bound (LeftAuthority44430.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44431RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44430.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44430.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound44473.bound LeftAuthority44430.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44473.bound, LeftAuthority44430.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound44473.actual selector witness) * (LeftAuthority44430.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44477

namespace LeftBound44488
def owner : Owner := ⟨.program ⟨214⟩, ⟨14802⟩⟩
def transferEvent : Nat := 44488
def frameStart : Nat := 44386
def rule : BoundRule := .product (.predecessor 0 44486 .coefficient) (.predecessor 1 44487 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44486 .coefficient)
      LeftAuthority44441.bound (LeftAuthority44441.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44442RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44441.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44441.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44487 .coefficient)
      LeftAuthority44484.bound (LeftAuthority44484.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44485RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44484.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44484.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority44441.bound LeftAuthority44484.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44441.bound, LeftAuthority44484.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority44441.actual selector witness) * (LeftAuthority44484.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44488

namespace LeftBound44496
def owner : Owner := ⟨.program ⟨214⟩, ⟨14803⟩⟩
def transferEvent : Nat := 44496
def frameStart : Nat := 44386
def rule : BoundRule := .sum [.predecessor 0 44494 .coefficient, .predecessor 1 44495 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44494 .coefficient)
      LeftAuthority44492.bound (LeftAuthority44492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44493RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44492.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44492.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44495 .coefficient)
      LeftBound44488.bound (LeftBound44488.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44488.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44488.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority44492.bound, LeftBound44488.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44492.bound, LeftBound44488.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority44492.actual selector witness, LeftBound44488.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44496

namespace LeftBound44500
def owner : Owner := ⟨.program ⟨214⟩, ⟨24925⟩⟩
def transferEvent : Nat := 44500
def frameStart : Nat := 44386
def rule : BoundRule := .sum [.predecessor 0 44498 .coefficient, .predecessor 1 44499 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44498 .coefficient)
      LeftBound44496.bound (LeftBound44496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44497RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44496.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44496.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44499 .coefficient)
      LeftBound44477.bound (LeftBound44477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44477.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44496.bound, LeftBound44477.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44496.bound, LeftBound44477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44496.actual selector witness, LeftBound44477.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44500

namespace LeftBound44513
def owner : Owner := ⟨.program ⟨214⟩, ⟨24923⟩⟩
def transferEvent : Nat := 44513
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44511 .coefficient, .predecessor 1 44512 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44511 .coefficient)
      LeftBound44334.bound (LeftBound44334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44334.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44334.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44512 .coefficient)
      LeftBound44317.bound (LeftBound44317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44324RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44317.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44317.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44334.bound, LeftBound44317.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44334.bound, LeftBound44317.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44334.actual selector witness, LeftBound44317.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44513

namespace LeftBound44516
def owner : Owner := ⟨.program ⟨214⟩, ⟨24923⟩⟩
def transferEvent : Nat := 44516
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44510 .summary, .result 44324 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44510 .summary)
      LeftBound44336.bound (LeftBound44336.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19035⟩⟩) (rawTerms := some (Proof.Events173.exact44510RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44336.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44324 .summary)
      LeftBound44319.bound (LeftBound44319.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24922⟩⟩) (rawTerms := some (Proof.Events173.exact44324RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44319.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44336.bound, LeftBound44319.bound]
def bound : CoeffClass := .finite ⟨352011863863296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44336.bound, LeftBound44319.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44336.actual selector witness, LeftBound44319.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44516

namespace LeftBound44520
def owner : Owner := ⟨.program ⟨214⟩, ⟨26384⟩⟩
def transferEvent : Nat := 44520
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 44518 .coefficient) (.predecessor 1 44519 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44518 .coefficient)
      LeftBound44513.bound (LeftBound44513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44513.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44513.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44519 .coefficient)
      LeftAuthority44239.bound (LeftAuthority44239.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44239.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44239.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound44513.bound LeftAuthority44239.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44513.bound, LeftAuthority44239.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound44513.actual selector witness) * (LeftAuthority44239.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44520

namespace LeftBound44521
def owner : Owner := ⟨.program ⟨214⟩, ⟨26384⟩⟩
def transferEvent : Nat := 44521
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26382⟩⟩]⟩ [⟨.result 44240 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44240 .coefficient)
      LeftAuthority44239.bound (LeftAuthority44239.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26382⟩⟩) (rawTerms := some (Proof.Events172.exact44240RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44239.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44239.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority44239.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44239.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority44239.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound44521

namespace LeftBound44522
def owner : Owner := ⟨.program ⟨214⟩, ⟨26384⟩⟩
def transferEvent : Nat := 44522
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 44517 .summary) (.transfer 44521) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44517 .summary)
      LeftBound44516.bound (LeftBound44516.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24923⟩⟩) (rawTerms := some (Proof.Events173.exact44517RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44516.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 44521)
      LeftBound44521.bound (LeftBound44521.actual selector witness) := by
  exact .transfer (LeftBound44521.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound44516.bound LeftBound44521.bound
def bound : CoeffClass := .finite ⟨1291889172568118132736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44516.bound, LeftBound44521.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound44516.actual selector witness) * (LeftBound44521.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44522

namespace LeftBound44533
def owner : Owner := ⟨.program ⟨214⟩, ⟨20402⟩⟩
def transferEvent : Nat := 44533
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 44531 .coefficient) (.value (.predecessor 1 44532 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44531 .coefficient)
      LeftAuthority44529.bound (LeftAuthority44529.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44529.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44529.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44532 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority44529.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44529.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority44529.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound44533

namespace LeftBound44537
def owner : Owner := ⟨.program ⟨214⟩, ⟨20403⟩⟩
def transferEvent : Nat := 44537
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 44535 .coefficient) (.predecessor 1 44536 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44535 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44536 .coefficient)
      LeftBound44533.bound (LeftBound44533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44534RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44533.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44533.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound44533.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound44533.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound44533.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44537

namespace LeftBound44538
def owner : Owner := ⟨.program ⟨214⟩, ⟨20403⟩⟩
def transferEvent : Nat := 44538
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20400⟩⟩]⟩ [⟨.result 44530 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44530 .coefficient)
      LeftAuthority44529.bound (LeftAuthority44529.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20400⟩⟩) (rawTerms := some (Proof.Events173.exact44530RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44529.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44529.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority44529.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44529.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority44529.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound44538

namespace LeftBound44539
def owner : Owner := ⟨.program ⟨214⟩, ⟨20403⟩⟩
def transferEvent : Nat := 44539
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 44538) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 44538)
      LeftBound44538.bound (LeftBound44538.actual selector witness) := by
  exact .transfer (LeftBound44538.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound44538.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound44538.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound44538.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44539

namespace LeftBound44634
def owner : Owner := ⟨.program ⟨214⟩, ⟨14801⟩⟩
def transferEvent : Nat := 44634
def frameStart : Nat := 44595
def rule : BoundRule := .identity (.predecessor 0 44633 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44633 .coefficient)
      LeftAuthority44631.bound (LeftAuthority44631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44632RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44631.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44631.derived selector witness)

def rawBound : CoeffClass := LeftAuthority44631.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority44631.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound44634

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
