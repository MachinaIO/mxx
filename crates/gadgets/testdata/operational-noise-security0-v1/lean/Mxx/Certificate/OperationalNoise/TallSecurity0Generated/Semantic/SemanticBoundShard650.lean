import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard649

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound95496
def owner : Owner := ⟨.program ⟨214⟩, ⟨29569⟩⟩
def transferEvent : Nat := 95496
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29567⟩⟩]⟩ [⟨.result 95239 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95239 .coefficient)
      LeftAuthority95238.bound (LeftAuthority95238.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29567⟩⟩) (rawTerms := some (Proof.Events372.exact95239RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95238.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95238.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority95238.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95238.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority95238.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound95496

namespace LeftBound95497
def owner : Owner := ⟨.program ⟨214⟩, ⟨29569⟩⟩
def transferEvent : Nat := 95497
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 95492 .summary) (.transfer 95496) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95492 .summary)
      LeftBound95491.bound (LeftBound95491.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25593⟩⟩) (rawTerms := some (Proof.Events373.exact95492RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95491.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 95496)
      LeftBound95496.bound (LeftBound95496.actual selector witness) := by
  exact .transfer (LeftBound95496.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95491.bound LeftBound95496.bound
def bound : CoeffClass := .finite ⟨1292449483693632782336, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95491.bound, LeftBound95496.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95491.actual selector witness) * (LeftBound95496.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95497

namespace LeftBound95508
def owner : Owner := ⟨.program ⟨214⟩, ⟨22543⟩⟩
def transferEvent : Nat := 95508
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 95506 .coefficient) (.value (.predecessor 1 95507 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95506 .coefficient)
      LeftAuthority95504.bound (LeftAuthority95504.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95505RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95504.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95504.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95507 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority95504.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95504.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority95504.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound95508

namespace LeftBound95512
def owner : Owner := ⟨.program ⟨214⟩, ⟨22544⟩⟩
def transferEvent : Nat := 95512
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 95510 .coefficient) (.predecessor 1 95511 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95510 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95511 .coefficient)
      LeftBound95508.bound (LeftBound95508.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95509RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95508.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound95508.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound95508.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound95508.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95512

namespace LeftBound95513
def owner : Owner := ⟨.program ⟨214⟩, ⟨22544⟩⟩
def transferEvent : Nat := 95513
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22541⟩⟩]⟩ [⟨.result 95505 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95505 .coefficient)
      LeftAuthority95504.bound (LeftAuthority95504.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22541⟩⟩) (rawTerms := some (Proof.Events373.exact95505RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95504.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95504.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority95504.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95504.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority95504.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound95513

namespace LeftBound95514
def owner : Owner := ⟨.program ⟨214⟩, ⟨22544⟩⟩
def transferEvent : Nat := 95514
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 95513) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 95513)
      LeftBound95513.bound (LeftBound95513.actual selector witness) := by
  exact .transfer (LeftBound95513.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound95513.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound95513.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound95513.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95514

namespace LeftBound95585
def owner : Owner := ⟨.program ⟨214⟩, ⟨16743⟩⟩
def transferEvent : Nat := 95585
def frameStart : Nat := 95558
def rule : BoundRule := .identity (.predecessor 0 95584 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95584 .coefficient)
      LeftAuthority95582.bound (LeftAuthority95582.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95582.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95582.derived selector witness)

def rawBound : CoeffClass := LeftAuthority95582.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority95582.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound95585

namespace LeftBound95602
def owner : Owner := ⟨.program ⟨214⟩, ⟨16819⟩⟩
def transferEvent : Nat := 95602
def frameStart : Nat := 95558
def rule : BoundRule := .sum [.predecessor 0 95600 .coefficient, .predecessor 1 95601 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95600 .coefficient)
      LeftBound95585.bound (LeftBound95585.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound95585.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95601 .coefficient)
      LeftAuthority95598.bound (LeftAuthority95598.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority95598.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95585.bound, LeftAuthority95598.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95585.bound, LeftAuthority95598.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95585.actual selector witness, LeftAuthority95598.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95602

namespace LeftBound95605
def owner : Owner := ⟨.program ⟨214⟩, ⟨16820⟩⟩
def transferEvent : Nat := 95605
def frameStart : Nat := 95558
def rule : BoundRule := .identity (.predecessor 0 95604 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95604 .coefficient)
      LeftBound95602.bound (LeftBound95602.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound95602.derived selector witness)

def rawBound : CoeffClass := LeftBound95602.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95602.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound95602.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound95605

namespace LeftBound95611
def owner : Owner := ⟨.program ⟨214⟩, ⟨16821⟩⟩
def transferEvent : Nat := 95611
def frameStart : Nat := 95558
def rule : BoundRule := .product (.predecessor 0 95609 .coefficient) (.predecessor 1 95610 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95609 .coefficient)
      LeftAuthority95607.bound (LeftAuthority95607.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95607.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95607.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95610 .coefficient)
      LeftBound95605.bound (LeftBound95605.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95606RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95605.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95605.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority95607.bound LeftBound95605.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95607.bound, LeftBound95605.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority95607.actual selector witness) * (LeftBound95605.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95611

namespace LeftBound95619
def owner : Owner := ⟨.program ⟨214⟩, ⟨16822⟩⟩
def transferEvent : Nat := 95619
def frameStart : Nat := 95558
def rule : BoundRule := .sum [.predecessor 0 95617 .coefficient, .predecessor 1 95618 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95617 .coefficient)
      LeftAuthority95615.bound (LeftAuthority95615.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95615.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95615.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95618 .coefficient)
      LeftBound95611.bound (LeftBound95611.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95613RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95611.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95611.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority95615.bound, LeftBound95611.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95615.bound, LeftBound95611.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority95615.actual selector witness, LeftBound95611.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95619

namespace LeftBound95623
def owner : Owner := ⟨.program ⟨214⟩, ⟨29568⟩⟩
def transferEvent : Nat := 95623
def frameStart : Nat := 95558
def rule : BoundRule := .product (.predecessor 0 95621 .coefficient) (.predecessor 1 95622 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95621 .coefficient)
      LeftBound95619.bound (LeftBound95619.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95619.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95619.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95622 .coefficient)
      LeftAuthority95596.bound (LeftAuthority95596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95597RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95596.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95596.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95619.bound LeftAuthority95596.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95619.bound, LeftAuthority95596.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95619.actual selector witness) * (LeftAuthority95596.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95623

namespace LeftBound95634
def owner : Owner := ⟨.program ⟨214⟩, ⟨16792⟩⟩
def transferEvent : Nat := 95634
def frameStart : Nat := 95558
def rule : BoundRule := .product (.predecessor 0 95632 .coefficient) (.predecessor 1 95633 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95632 .coefficient)
      LeftAuthority95607.bound (LeftAuthority95607.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95607.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95607.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95633 .coefficient)
      LeftAuthority95630.bound (LeftAuthority95630.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95631RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95630.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95630.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority95607.bound LeftAuthority95630.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95607.bound, LeftAuthority95630.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority95607.actual selector witness) * (LeftAuthority95630.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95634

namespace LeftBound95642
def owner : Owner := ⟨.program ⟨214⟩, ⟨16793⟩⟩
def transferEvent : Nat := 95642
def frameStart : Nat := 95558
def rule : BoundRule := .sum [.predecessor 0 95640 .coefficient, .predecessor 1 95641 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95640 .coefficient)
      LeftAuthority95638.bound (LeftAuthority95638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95639RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95638.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95638.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95641 .coefficient)
      LeftBound95634.bound (LeftBound95634.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95634.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95634.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority95638.bound, LeftBound95634.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95638.bound, LeftBound95634.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority95638.actual selector witness, LeftBound95634.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95642

namespace LeftBound95646
def owner : Owner := ⟨.program ⟨214⟩, ⟨29572⟩⟩
def transferEvent : Nat := 95646
def frameStart : Nat := 95558
def rule : BoundRule := .sum [.predecessor 0 95644 .coefficient, .predecessor 1 95645 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95644 .coefficient)
      LeftBound95642.bound (LeftBound95642.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95643RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95642.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95642.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95645 .coefficient)
      LeftBound95623.bound (LeftBound95623.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95623.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95623.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95642.bound, LeftBound95623.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95642.bound, LeftBound95623.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95642.actual selector witness, LeftBound95623.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95646

namespace LeftBound95659
def owner : Owner := ⟨.program ⟨214⟩, ⟨29570⟩⟩
def transferEvent : Nat := 95659
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 95657 .coefficient, .predecessor 1 95658 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95657 .coefficient)
      LeftBound95512.bound (LeftBound95512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95656RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95512.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95658 .coefficient)
      LeftBound95495.bound (LeftBound95495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95502RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95495.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95495.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95512.bound, LeftBound95495.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95512.bound, LeftBound95495.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95512.actual selector witness, LeftBound95495.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95659

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
