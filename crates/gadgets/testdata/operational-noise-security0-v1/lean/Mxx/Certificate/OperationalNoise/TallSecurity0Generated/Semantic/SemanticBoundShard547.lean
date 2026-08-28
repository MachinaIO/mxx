import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard546

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound80597
def owner : Owner := ⟨.program ⟨214⟩, ⟨13250⟩⟩
def transferEvent : Nat := 80597
def frameStart : Nat := 80547
def rule : BoundRule := .sum [.predecessor 0 80595 .coefficient, .predecessor 1 80596 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80595 .coefficient)
      LeftBound80580.bound (LeftBound80580.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound80580.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80596 .coefficient)
      LeftAuthority80593.bound (LeftAuthority80593.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority80593.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80580.bound, LeftAuthority80593.bound]
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80580.bound, LeftAuthority80593.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80580.actual selector witness, LeftAuthority80593.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80597

namespace LeftBound80600
def owner : Owner := ⟨.program ⟨214⟩, ⟨13251⟩⟩
def transferEvent : Nat := 80600
def frameStart : Nat := 80547
def rule : BoundRule := .identity (.predecessor 0 80599 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80599 .coefficient)
      LeftBound80597.bound (LeftBound80597.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound80597.derived selector witness)

def rawBound : CoeffClass := LeftBound80597.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80597.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound80597.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound80600

namespace LeftBound80606
def owner : Owner := ⟨.program ⟨214⟩, ⟨13252⟩⟩
def transferEvent : Nat := 80606
def frameStart : Nat := 80547
def rule : BoundRule := .product (.predecessor 0 80604 .coefficient) (.predecessor 1 80605 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80604 .coefficient)
      LeftAuthority80602.bound (LeftAuthority80602.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80603RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80602.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80602.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80605 .coefficient)
      LeftBound80600.bound (LeftBound80600.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80601RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80600.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80600.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority80602.bound LeftBound80600.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80602.bound, LeftBound80600.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority80602.actual selector witness) * (LeftBound80600.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80606

namespace LeftBound80620
def owner : Owner := ⟨.program ⟨214⟩, ⟨7880⟩⟩
def transferEvent : Nat := 80620
def frameStart : Nat := 80547
def rule : BoundRule := .scale (.predecessor 0 80618 .coefficient) (.value (.predecessor 1 80619 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80618 .coefficient)
      LeftAuthority80616.bound (LeftAuthority80616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80617RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80616.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80616.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80619 .coefficient)
      LeftAuthority80550.bound (LeftAuthority80550.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority80550.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority80616.bound LeftAuthority80550.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80616.bound, LeftAuthority80550.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority80616.actual selector witness) * (LeftAuthority80550.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound80620

namespace LeftBound80623
def owner : Owner := ⟨.program ⟨214⟩, ⟨6769⟩⟩
def transferEvent : Nat := 80623
def frameStart : Nat := 80547
def rule : BoundRule := .identity (.predecessor 0 80622 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80622 .coefficient)
      LeftAuthority80610.bound (LeftAuthority80610.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80610.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80610.derived selector witness)

def rawBound : CoeffClass := LeftAuthority80610.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority80610.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound80623

namespace LeftBound80627
def owner : Owner := ⟨.program ⟨214⟩, ⟨7881⟩⟩
def transferEvent : Nat := 80627
def frameStart : Nat := 80547
def rule : BoundRule := .product (.predecessor 0 80625 .coefficient) (.predecessor 1 80626 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80625 .coefficient)
      LeftBound80623.bound (LeftBound80623.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80624RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80623.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80623.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80626 .coefficient)
      LeftBound80620.bound (LeftBound80620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80621RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80620.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80620.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound80623.bound LeftBound80620.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80623.bound, LeftBound80620.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound80623.actual selector witness) * (LeftBound80620.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80627

namespace LeftBound80632
def owner : Owner := ⟨.program ⟨214⟩, ⟨13253⟩⟩
def transferEvent : Nat := 80632
def frameStart : Nat := 80547
def rule : BoundRule := .sum [.predecessor 0 80630 .coefficient, .predecessor 1 80631 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80630 .coefficient)
      LeftBound80627.bound (LeftBound80627.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80629RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80627.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80627.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80631 .coefficient)
      LeftBound80606.bound (LeftBound80606.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80606.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80606.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80627.bound, LeftBound80606.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80627.bound, LeftBound80606.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80627.actual selector witness, LeftBound80606.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80632

namespace LeftBound80636
def owner : Owner := ⟨.program ⟨214⟩, ⟨25684⟩⟩
def transferEvent : Nat := 80636
def frameStart : Nat := 80547
def rule : BoundRule := .product (.predecessor 0 80634 .coefficient) (.predecessor 1 80635 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80634 .coefficient)
      LeftBound80632.bound (LeftBound80632.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80633RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80632.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80632.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80635 .coefficient)
      LeftAuthority80591.bound (LeftAuthority80591.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80591.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80591.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound80632.bound LeftAuthority80591.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80632.bound, LeftAuthority80591.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound80632.actual selector witness) * (LeftAuthority80591.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80636

namespace LeftBound80647
def owner : Owner := ⟨.program ⟨214⟩, ⟨16873⟩⟩
def transferEvent : Nat := 80647
def frameStart : Nat := 80547
def rule : BoundRule := .product (.predecessor 0 80645 .coefficient) (.predecessor 1 80646 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80645 .coefficient)
      LeftAuthority80602.bound (LeftAuthority80602.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80603RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80602.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80602.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80646 .coefficient)
      LeftAuthority80643.bound (LeftAuthority80643.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80644RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80643.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80643.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority80602.bound LeftAuthority80643.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80602.bound, LeftAuthority80643.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority80602.actual selector witness) * (LeftAuthority80643.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80647

namespace LeftBound80655
def owner : Owner := ⟨.program ⟨214⟩, ⟨16874⟩⟩
def transferEvent : Nat := 80655
def frameStart : Nat := 80547
def rule : BoundRule := .sum [.predecessor 0 80653 .coefficient, .predecessor 1 80654 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80653 .coefficient)
      LeftAuthority80651.bound (LeftAuthority80651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80652RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80651.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80651.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80654 .coefficient)
      LeftBound80647.bound (LeftBound80647.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80649RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80647.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80647.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority80651.bound, LeftBound80647.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80651.bound, LeftBound80647.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority80651.actual selector witness, LeftBound80647.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80655

namespace LeftBound80659
def owner : Owner := ⟨.program ⟨214⟩, ⟨25685⟩⟩
def transferEvent : Nat := 80659
def frameStart : Nat := 80547
def rule : BoundRule := .sum [.predecessor 0 80657 .coefficient, .predecessor 1 80658 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80657 .coefficient)
      LeftBound80655.bound (LeftBound80655.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80656RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80655.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80655.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80658 .coefficient)
      LeftBound80636.bound (LeftBound80636.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80641RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80636.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80636.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80655.bound, LeftBound80636.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80655.bound, LeftBound80636.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80655.actual selector witness, LeftBound80636.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80659

namespace LeftBound80672
def owner : Owner := ⟨.program ⟨214⟩, ⟨25683⟩⟩
def transferEvent : Nat := 80672
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 80670 .coefficient, .predecessor 1 80671 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80670 .coefficient)
      LeftBound80495.bound (LeftBound80495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80669RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80495.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80495.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80671 .coefficient)
      LeftBound80478.bound (LeftBound80478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80485RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80478.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80478.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80495.bound, LeftBound80478.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80495.bound, LeftBound80478.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80495.actual selector witness, LeftBound80478.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80672

namespace LeftBound80675
def owner : Owner := ⟨.program ⟨214⟩, ⟨25683⟩⟩
def transferEvent : Nat := 80675
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 80669 .summary, .result 80485 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80669 .summary)
      LeftBound80497.bound (LeftBound80497.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20179⟩⟩) (rawTerms := some (Proof.Events315.exact80669RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80497.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80485 .summary)
      LeftBound80480.bound (LeftBound80480.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25682⟩⟩) (rawTerms := some (Proof.Events314.exact80485RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80480.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80497.bound, LeftBound80480.bound]
def bound : CoeffClass := .finite ⟨352182857248768, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80497.bound, LeftBound80480.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80497.actual selector witness, LeftBound80480.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80675

namespace LeftBound80679
def owner : Owner := ⟨.program ⟨214⟩, ⟨29821⟩⟩
def transferEvent : Nat := 80679
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 80677 .coefficient) (.predecessor 1 80678 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80677 .coefficient)
      LeftBound80672.bound (LeftBound80672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80676RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80672.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80672.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80678 .coefficient)
      LeftAuthority80400.bound (LeftAuthority80400.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events314.exact80401RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80400.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80400.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound80672.bound LeftAuthority80400.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80672.bound, LeftAuthority80400.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound80672.actual selector witness) * (LeftAuthority80400.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80679

namespace LeftBound80680
def owner : Owner := ⟨.program ⟨214⟩, ⟨29821⟩⟩
def transferEvent : Nat := 80680
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29819⟩⟩]⟩ [⟨.result 80401 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80401 .coefficient)
      LeftAuthority80400.bound (LeftAuthority80400.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29819⟩⟩) (rawTerms := some (Proof.Events314.exact80401RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80400.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80400.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority80400.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80400.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority80400.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound80680

namespace LeftBound80681
def owner : Owner := ⟨.program ⟨214⟩, ⟨29821⟩⟩
def transferEvent : Nat := 80681
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80676 .summary) (.transfer 80680) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80676 .summary)
      LeftBound80675.bound (LeftBound80675.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25683⟩⟩) (rawTerms := some (Proof.Events315.exact80676RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80675.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 80680)
      LeftBound80680.bound (LeftBound80680.actual selector witness) := by
  exact .transfer (LeftBound80680.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound80675.bound LeftBound80680.bound
def bound : CoeffClass := .finite ⟨1292516721028694540288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80675.bound, LeftBound80680.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound80675.actual selector witness) * (LeftBound80680.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80681

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
