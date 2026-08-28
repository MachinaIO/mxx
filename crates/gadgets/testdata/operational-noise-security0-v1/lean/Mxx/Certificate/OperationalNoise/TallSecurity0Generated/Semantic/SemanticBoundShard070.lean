import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard069

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound11576
def owner : Owner := ⟨.program ⟨214⟩, ⟨19547⟩⟩
def transferEvent : Nat := 11576
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 11575) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 11575)
      LeftBound11575.bound (LeftBound11575.actual selector witness) := by
  exact .transfer (LeftBound11575.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound11575.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound11575.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound11575.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11576

namespace LeftBound11655
def owner : Owner := ⟨.program ⟨214⟩, ⟨14244⟩⟩
def transferEvent : Nat := 11655
def frameStart : Nat := 11626
def rule : BoundRule := .product (.predecessor 0 11653 .coefficient) (.predecessor 1 11654 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11653 .coefficient)
      LeftAuthority11651.bound (LeftAuthority11651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11652RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11651.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11651.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11654 .coefficient)
      LeftAuthority11648.bound (LeftAuthority11648.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11649RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11648.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11648.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority11651.bound LeftAuthority11648.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11651.bound, LeftAuthority11648.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority11651.actual selector witness) * (LeftAuthority11648.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11655

namespace LeftBound11659
def owner : Owner := ⟨.program ⟨214⟩, ⟨14245⟩⟩
def transferEvent : Nat := 11659
def frameStart : Nat := 11626
def rule : BoundRule := .identity (.predecessor 0 11658 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11658 .coefficient)
      LeftBound11655.bound (LeftBound11655.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11657RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11655.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11655.derived selector witness)

def rawBound : CoeffClass := LeftBound11655.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11655.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound11655.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound11659

namespace LeftBound11676
def owner : Owner := ⟨.program ⟨214⟩, ⟨14330⟩⟩
def transferEvent : Nat := 11676
def frameStart : Nat := 11626
def rule : BoundRule := .sum [.predecessor 0 11674 .coefficient, .predecessor 1 11675 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11674 .coefficient)
      LeftBound11659.bound (LeftBound11659.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound11659.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11675 .coefficient)
      LeftAuthority11672.bound (LeftAuthority11672.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority11672.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11659.bound, LeftAuthority11672.bound]
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11659.bound, LeftAuthority11672.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11659.actual selector witness, LeftAuthority11672.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11676

namespace LeftBound11679
def owner : Owner := ⟨.program ⟨214⟩, ⟨14331⟩⟩
def transferEvent : Nat := 11679
def frameStart : Nat := 11626
def rule : BoundRule := .identity (.predecessor 0 11678 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11678 .coefficient)
      LeftBound11676.bound (LeftBound11676.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound11676.derived selector witness)

def rawBound : CoeffClass := LeftBound11676.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11676.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound11676.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound11679

namespace LeftBound11685
def owner : Owner := ⟨.program ⟨214⟩, ⟨14332⟩⟩
def transferEvent : Nat := 11685
def frameStart : Nat := 11626
def rule : BoundRule := .product (.predecessor 0 11683 .coefficient) (.predecessor 1 11684 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11683 .coefficient)
      LeftAuthority11681.bound (LeftAuthority11681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11682RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11681.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11681.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11684 .coefficient)
      LeftBound11679.bound (LeftBound11679.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11680RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11679.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11679.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority11681.bound LeftBound11679.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11681.bound, LeftBound11679.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority11681.actual selector witness) * (LeftBound11679.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11685

namespace LeftBound11701
def owner : Owner := ⟨.program ⟨214⟩, ⟨7853⟩⟩
def transferEvent : Nat := 11701
def frameStart : Nat := 11626
def rule : BoundRule := .scale (.predecessor 0 11699 .coefficient) (.value (.predecessor 1 11700 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11699 .coefficient)
      LeftAuthority11697.bound (LeftAuthority11697.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11698RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11697.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11697.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11700 .coefficient)
      LeftAuthority11688.bound (LeftAuthority11688.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority11688.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority11697.bound LeftAuthority11688.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11697.bound, LeftAuthority11688.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority11697.actual selector witness) * (LeftAuthority11688.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound11701

namespace LeftBound11704
def owner : Owner := ⟨.program ⟨214⟩, ⟨6759⟩⟩
def transferEvent : Nat := 11704
def frameStart : Nat := 11626
def rule : BoundRule := .identity (.predecessor 0 11703 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11703 .coefficient)
      LeftAuthority11691.bound (LeftAuthority11691.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11692RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11691.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11691.derived selector witness)

def rawBound : CoeffClass := LeftAuthority11691.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11691.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority11691.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound11704

namespace LeftBound11708
def owner : Owner := ⟨.program ⟨214⟩, ⟨7854⟩⟩
def transferEvent : Nat := 11708
def frameStart : Nat := 11626
def rule : BoundRule := .product (.predecessor 0 11706 .coefficient) (.predecessor 1 11707 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11706 .coefficient)
      LeftBound11704.bound (LeftBound11704.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11705RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11704.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11704.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11707 .coefficient)
      LeftBound11701.bound (LeftBound11701.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11702RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11701.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11701.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound11704.bound LeftBound11701.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11704.bound, LeftBound11701.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound11704.actual selector witness) * (LeftBound11701.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11708

namespace LeftBound11713
def owner : Owner := ⟨.program ⟨214⟩, ⟨14333⟩⟩
def transferEvent : Nat := 11713
def frameStart : Nat := 11626
def rule : BoundRule := .sum [.predecessor 0 11711 .coefficient, .predecessor 1 11712 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11711 .coefficient)
      LeftBound11708.bound (LeftBound11708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11710RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11708.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11708.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11712 .coefficient)
      LeftBound11685.bound (LeftBound11685.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11687RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11685.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11685.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11708.bound, LeftBound11685.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11708.bound, LeftBound11685.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11708.actual selector witness, LeftBound11685.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11713

namespace LeftBound11717
def owner : Owner := ⟨.program ⟨214⟩, ⟨26089⟩⟩
def transferEvent : Nat := 11717
def frameStart : Nat := 11626
def rule : BoundRule := .product (.predecessor 0 11715 .coefficient) (.predecessor 1 11716 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11715 .coefficient)
      LeftBound11713.bound (LeftBound11713.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11714RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11713.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11713.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11716 .coefficient)
      LeftAuthority11670.bound (LeftAuthority11670.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11671RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11670.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11670.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound11713.bound LeftAuthority11670.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11713.bound, LeftAuthority11670.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound11713.actual selector witness) * (LeftAuthority11670.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11717

namespace LeftBound11728
def owner : Owner := ⟨.program ⟨214⟩, ⟨15958⟩⟩
def transferEvent : Nat := 11728
def frameStart : Nat := 11626
def rule : BoundRule := .product (.predecessor 0 11726 .coefficient) (.predecessor 1 11727 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11726 .coefficient)
      LeftAuthority11681.bound (LeftAuthority11681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11682RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11681.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11681.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11727 .coefficient)
      LeftAuthority11724.bound (LeftAuthority11724.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11724.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11724.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority11681.bound LeftAuthority11724.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11681.bound, LeftAuthority11724.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority11681.actual selector witness) * (LeftAuthority11724.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11728

namespace LeftBound11736
def owner : Owner := ⟨.program ⟨214⟩, ⟨15959⟩⟩
def transferEvent : Nat := 11736
def frameStart : Nat := 11626
def rule : BoundRule := .sum [.predecessor 0 11734 .coefficient, .predecessor 1 11735 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11734 .coefficient)
      LeftAuthority11732.bound (LeftAuthority11732.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11733RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11732.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11732.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11735 .coefficient)
      LeftBound11728.bound (LeftBound11728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11730RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11728.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11728.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority11732.bound, LeftBound11728.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11732.bound, LeftBound11728.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority11732.actual selector witness, LeftBound11728.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11736

namespace LeftBound11740
def owner : Owner := ⟨.program ⟨214⟩, ⟨26090⟩⟩
def transferEvent : Nat := 11740
def frameStart : Nat := 11626
def rule : BoundRule := .sum [.predecessor 0 11738 .coefficient, .predecessor 1 11739 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11738 .coefficient)
      LeftBound11736.bound (LeftBound11736.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11737RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11736.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11736.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11739 .coefficient)
      LeftBound11717.bound (LeftBound11717.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11722RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11717.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11717.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11736.bound, LeftBound11717.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11736.bound, LeftBound11717.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11736.actual selector witness, LeftBound11717.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11740

namespace LeftBound11753
def owner : Owner := ⟨.program ⟨214⟩, ⟨26088⟩⟩
def transferEvent : Nat := 11753
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11751 .coefficient, .predecessor 1 11752 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11751 .coefficient)
      LeftBound11574.bound (LeftBound11574.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11574.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11574.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11752 .coefficient)
      LeftBound11557.bound (LeftBound11557.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11564RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11557.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11557.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11574.bound, LeftBound11557.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11574.bound, LeftBound11557.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11574.actual selector witness, LeftBound11557.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11753

namespace LeftBound11756
def owner : Owner := ⟨.program ⟨214⟩, ⟨26088⟩⟩
def transferEvent : Nat := 11756
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 11750 .summary, .result 11564 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11750 .summary)
      LeftBound11576.bound (LeftBound11576.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19547⟩⟩) (rawTerms := some (Proof.Events045.exact11750RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound11576.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11564 .summary)
      LeftBound11559.bound (LeftBound11559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26087⟩⟩) (rawTerms := some (Proof.Events045.exact11564RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound11559.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11576.bound, LeftBound11559.bound]
def bound : CoeffClass := .finite ⟨352060719116288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11576.bound, LeftBound11559.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11576.actual selector witness, LeftBound11559.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11756

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
