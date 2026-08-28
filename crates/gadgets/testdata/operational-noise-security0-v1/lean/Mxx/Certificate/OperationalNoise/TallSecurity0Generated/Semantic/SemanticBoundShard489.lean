import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard488

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound71741
def owner : Owner := ⟨.program ⟨214⟩, ⟨13549⟩⟩
def transferEvent : Nat := 71741
def frameStart : Nat := 71708
def rule : BoundRule := .identity (.predecessor 0 71740 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71740 .coefficient)
      LeftBound71737.bound (LeftBound71737.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71739RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71737.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71737.derived selector witness)

def rawBound : CoeffClass := LeftBound71737.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71737.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound71737.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound71741

namespace LeftBound71758
def owner : Owner := ⟨.program ⟨214⟩, ⟨13659⟩⟩
def transferEvent : Nat := 71758
def frameStart : Nat := 71708
def rule : BoundRule := .sum [.predecessor 0 71756 .coefficient, .predecessor 1 71757 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71756 .coefficient)
      LeftBound71741.bound (LeftBound71741.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound71741.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71757 .coefficient)
      LeftAuthority71754.bound (LeftAuthority71754.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority71754.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71741.bound, LeftAuthority71754.bound]
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71741.bound, LeftAuthority71754.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71741.actual selector witness, LeftAuthority71754.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71758

namespace LeftBound71761
def owner : Owner := ⟨.program ⟨214⟩, ⟨13660⟩⟩
def transferEvent : Nat := 71761
def frameStart : Nat := 71708
def rule : BoundRule := .identity (.predecessor 0 71760 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71760 .coefficient)
      LeftBound71758.bound (LeftBound71758.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound71758.derived selector witness)

def rawBound : CoeffClass := LeftBound71758.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71758.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound71758.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound71761

namespace LeftBound71767
def owner : Owner := ⟨.program ⟨214⟩, ⟨13661⟩⟩
def transferEvent : Nat := 71767
def frameStart : Nat := 71708
def rule : BoundRule := .product (.predecessor 0 71765 .coefficient) (.predecessor 1 71766 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71765 .coefficient)
      LeftAuthority71763.bound (LeftAuthority71763.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71764RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71763.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71763.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71766 .coefficient)
      LeftBound71761.bound (LeftBound71761.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71761.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71761.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority71763.bound LeftBound71761.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71763.bound, LeftBound71761.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority71763.actual selector witness) * (LeftBound71761.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71767

namespace LeftBound71783
def owner : Owner := ⟨.program ⟨214⟩, ⟨7844⟩⟩
def transferEvent : Nat := 71783
def frameStart : Nat := 71708
def rule : BoundRule := .scale (.predecessor 0 71781 .coefficient) (.value (.predecessor 1 71782 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71781 .coefficient)
      LeftAuthority71779.bound (LeftAuthority71779.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71779.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71779.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71782 .coefficient)
      LeftAuthority71770.bound (LeftAuthority71770.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority71770.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority71779.bound LeftAuthority71770.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71779.bound, LeftAuthority71770.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority71779.actual selector witness) * (LeftAuthority71770.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound71783

namespace LeftBound71786
def owner : Owner := ⟨.program ⟨214⟩, ⟨6793⟩⟩
def transferEvent : Nat := 71786
def frameStart : Nat := 71708
def rule : BoundRule := .identity (.predecessor 0 71785 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71785 .coefficient)
      LeftAuthority71773.bound (LeftAuthority71773.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71774RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71773.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71773.derived selector witness)

def rawBound : CoeffClass := LeftAuthority71773.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71773.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority71773.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound71786

namespace LeftBound71790
def owner : Owner := ⟨.program ⟨214⟩, ⟨7845⟩⟩
def transferEvent : Nat := 71790
def frameStart : Nat := 71708
def rule : BoundRule := .product (.predecessor 0 71788 .coefficient) (.predecessor 1 71789 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71788 .coefficient)
      LeftBound71786.bound (LeftBound71786.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71787RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71786.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71786.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71789 .coefficient)
      LeftBound71783.bound (LeftBound71783.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71784RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71783.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71783.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound71786.bound LeftBound71783.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71786.bound, LeftBound71783.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound71786.actual selector witness) * (LeftBound71783.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71790

namespace LeftBound71795
def owner : Owner := ⟨.program ⟨214⟩, ⟨13662⟩⟩
def transferEvent : Nat := 71795
def frameStart : Nat := 71708
def rule : BoundRule := .sum [.predecessor 0 71793 .coefficient, .predecessor 1 71794 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71793 .coefficient)
      LeftBound71790.bound (LeftBound71790.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71792RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71790.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71790.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71794 .coefficient)
      LeftBound71767.bound (LeftBound71767.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71769RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71767.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71767.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71790.bound, LeftBound71767.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71790.bound, LeftBound71767.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71790.actual selector witness, LeftBound71767.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71795

namespace LeftBound71799
def owner : Owner := ⟨.program ⟨214⟩, ⟨25833⟩⟩
def transferEvent : Nat := 71799
def frameStart : Nat := 71708
def rule : BoundRule := .product (.predecessor 0 71797 .coefficient) (.predecessor 1 71798 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71797 .coefficient)
      LeftBound71795.bound (LeftBound71795.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71796RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71795.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71795.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71798 .coefficient)
      LeftAuthority71752.bound (LeftAuthority71752.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71753RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71752.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71752.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound71795.bound LeftAuthority71752.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71795.bound, LeftAuthority71752.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound71795.actual selector witness) * (LeftAuthority71752.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71799

namespace LeftBound71810
def owner : Owner := ⟨.program ⟨214⟩, ⟨15581⟩⟩
def transferEvent : Nat := 71810
def frameStart : Nat := 71708
def rule : BoundRule := .product (.predecessor 0 71808 .coefficient) (.predecessor 1 71809 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71808 .coefficient)
      LeftAuthority71763.bound (LeftAuthority71763.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71764RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71763.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71763.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71809 .coefficient)
      LeftAuthority71806.bound (LeftAuthority71806.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71807RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71806.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71806.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority71763.bound LeftAuthority71806.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71763.bound, LeftAuthority71806.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority71763.actual selector witness) * (LeftAuthority71806.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71810

namespace LeftBound71818
def owner : Owner := ⟨.program ⟨214⟩, ⟨15582⟩⟩
def transferEvent : Nat := 71818
def frameStart : Nat := 71708
def rule : BoundRule := .sum [.predecessor 0 71816 .coefficient, .predecessor 1 71817 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71816 .coefficient)
      LeftAuthority71814.bound (LeftAuthority71814.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71814.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71814.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71817 .coefficient)
      LeftBound71810.bound (LeftBound71810.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71812RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71810.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71810.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority71814.bound, LeftBound71810.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71814.bound, LeftBound71810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority71814.actual selector witness, LeftBound71810.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71818

namespace LeftBound71822
def owner : Owner := ⟨.program ⟨214⟩, ⟨25834⟩⟩
def transferEvent : Nat := 71822
def frameStart : Nat := 71708
def rule : BoundRule := .sum [.predecessor 0 71820 .coefficient, .predecessor 1 71821 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71820 .coefficient)
      LeftBound71818.bound (LeftBound71818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71819RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71818.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71818.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71821 .coefficient)
      LeftBound71799.bound (LeftBound71799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71804RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71799.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71799.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71818.bound, LeftBound71799.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71818.bound, LeftBound71799.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71818.actual selector witness, LeftBound71799.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71822

namespace LeftBound71835
def owner : Owner := ⟨.program ⟨214⟩, ⟨25832⟩⟩
def transferEvent : Nat := 71835
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 71833 .coefficient, .predecessor 1 71834 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71833 .coefficient)
      LeftBound71656.bound (LeftBound71656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71832RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71656.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71656.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71834 .coefficient)
      LeftBound71639.bound (LeftBound71639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events279.exact71646RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71639.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71639.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71656.bound, LeftBound71639.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71656.bound, LeftBound71639.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71656.actual selector witness, LeftBound71639.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71835

namespace LeftBound71838
def owner : Owner := ⟨.program ⟨214⟩, ⟨25832⟩⟩
def transferEvent : Nat := 71838
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 71832 .summary, .result 71646 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71832 .summary)
      LeftBound71658.bound (LeftBound71658.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19311⟩⟩) (rawTerms := some (Proof.Events280.exact71832RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound71658.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71646 .summary)
      LeftBound71641.bound (LeftBound71641.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25831⟩⟩) (rawTerms := some (Proof.Events279.exact71646RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound71641.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71658.bound, LeftBound71641.bound]
def bound : CoeffClass := .finite ⟨352036291489792, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71658.bound, LeftBound71641.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71658.actual selector witness, LeftBound71641.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71838

namespace LeftBound71842
def owner : Owner := ⟨.program ⟨214⟩, ⟨27204⟩⟩
def transferEvent : Nat := 71842
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 71840 .coefficient) (.predecessor 1 71841 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71840 .coefficient)
      LeftBound71835.bound (LeftBound71835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71835.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71835.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71841 .coefficient)
      LeftAuthority71561.bound (LeftAuthority71561.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events279.exact71562RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71561.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71561.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound71835.bound LeftAuthority71561.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71835.bound, LeftAuthority71561.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound71835.actual selector witness) * (LeftAuthority71561.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71842

namespace LeftBound71843
def owner : Owner := ⟨.program ⟨214⟩, ⟨27204⟩⟩
def transferEvent : Nat := 71843
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27202⟩⟩]⟩ [⟨.result 71562 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71562 .coefficient)
      LeftAuthority71561.bound (LeftAuthority71561.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27202⟩⟩) (rawTerms := some (Proof.Events279.exact71562RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71561.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71561.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority71561.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71561.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority71561.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound71843

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
