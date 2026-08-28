import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard593

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound86816
def owner : Owner := ⟨.program ⟨214⟩, ⟨12164⟩⟩
def transferEvent : Nat := 86816
def frameStart : Nat := 86787
def rule : BoundRule := .product (.predecessor 0 86814 .coefficient) (.predecessor 1 86815 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86814 .coefficient)
      LeftAuthority86812.bound (LeftAuthority86812.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86813RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86812.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86812.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86815 .coefficient)
      LeftAuthority86809.bound (LeftAuthority86809.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86810RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86809.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86809.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority86812.bound LeftAuthority86809.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86812.bound, LeftAuthority86809.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority86812.actual selector witness) * (LeftAuthority86809.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86816

namespace LeftBound86820
def owner : Owner := ⟨.program ⟨214⟩, ⟨12165⟩⟩
def transferEvent : Nat := 86820
def frameStart : Nat := 86787
def rule : BoundRule := .identity (.predecessor 0 86819 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86819 .coefficient)
      LeftBound86816.bound (LeftBound86816.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86818RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86816.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86816.derived selector witness)

def rawBound : CoeffClass := LeftBound86816.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86816.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound86816.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound86820

namespace LeftBound86837
def owner : Owner := ⟨.program ⟨214⟩, ⟨12270⟩⟩
def transferEvent : Nat := 86837
def frameStart : Nat := 86787
def rule : BoundRule := .sum [.predecessor 0 86835 .coefficient, .predecessor 1 86836 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86835 .coefficient)
      LeftBound86820.bound (LeftBound86820.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound86820.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86836 .coefficient)
      LeftAuthority86833.bound (LeftAuthority86833.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority86833.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86820.bound, LeftAuthority86833.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86820.bound, LeftAuthority86833.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86820.actual selector witness, LeftAuthority86833.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86837

namespace LeftBound86840
def owner : Owner := ⟨.program ⟨214⟩, ⟨12271⟩⟩
def transferEvent : Nat := 86840
def frameStart : Nat := 86787
def rule : BoundRule := .identity (.predecessor 0 86839 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86839 .coefficient)
      LeftBound86837.bound (LeftBound86837.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound86837.derived selector witness)

def rawBound : CoeffClass := LeftBound86837.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86837.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound86837.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound86840

namespace LeftBound86846
def owner : Owner := ⟨.program ⟨214⟩, ⟨12272⟩⟩
def transferEvent : Nat := 86846
def frameStart : Nat := 86787
def rule : BoundRule := .product (.predecessor 0 86844 .coefficient) (.predecessor 1 86845 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86844 .coefficient)
      LeftAuthority86842.bound (LeftAuthority86842.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86843RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86842.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86842.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86845 .coefficient)
      LeftBound86840.bound (LeftBound86840.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86841RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86840.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86840.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority86842.bound LeftBound86840.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86842.bound, LeftBound86840.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority86842.actual selector witness) * (LeftBound86840.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86846

namespace LeftBound86860
def owner : Owner := ⟨.program ⟨214⟩, ⟨7841⟩⟩
def transferEvent : Nat := 86860
def frameStart : Nat := 86787
def rule : BoundRule := .scale (.predecessor 0 86858 .coefficient) (.value (.predecessor 1 86859 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86858 .coefficient)
      LeftAuthority86856.bound (LeftAuthority86856.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86857RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86856.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86856.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86859 .coefficient)
      LeftAuthority86790.bound (LeftAuthority86790.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority86790.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority86856.bound LeftAuthority86790.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86856.bound, LeftAuthority86790.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority86856.actual selector witness) * (LeftAuthority86790.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound86860

namespace LeftBound86863
def owner : Owner := ⟨.program ⟨214⟩, ⟨6792⟩⟩
def transferEvent : Nat := 86863
def frameStart : Nat := 86787
def rule : BoundRule := .identity (.predecessor 0 86862 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86862 .coefficient)
      LeftAuthority86850.bound (LeftAuthority86850.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86851RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86850.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86850.derived selector witness)

def rawBound : CoeffClass := LeftAuthority86850.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86850.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority86850.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound86863

namespace LeftBound86867
def owner : Owner := ⟨.program ⟨214⟩, ⟨7842⟩⟩
def transferEvent : Nat := 86867
def frameStart : Nat := 86787
def rule : BoundRule := .product (.predecessor 0 86865 .coefficient) (.predecessor 1 86866 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86865 .coefficient)
      LeftBound86863.bound (LeftBound86863.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86864RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86863.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86863.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86866 .coefficient)
      LeftBound86860.bound (LeftBound86860.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86861RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86860.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86860.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound86863.bound LeftBound86860.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86863.bound, LeftBound86860.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound86863.actual selector witness) * (LeftBound86860.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86867

namespace LeftBound86872
def owner : Owner := ⟨.program ⟨214⟩, ⟨12273⟩⟩
def transferEvent : Nat := 86872
def frameStart : Nat := 86787
def rule : BoundRule := .sum [.predecessor 0 86870 .coefficient, .predecessor 1 86871 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86870 .coefficient)
      LeftBound86867.bound (LeftBound86867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86867.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86867.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86871 .coefficient)
      LeftBound86846.bound (LeftBound86846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86848RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86846.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86846.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86867.bound, LeftBound86846.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86867.bound, LeftBound86846.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86867.actual selector witness, LeftBound86846.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86872

namespace LeftBound86876
def owner : Owner := ⟨.program ⟨214⟩, ⟨25299⟩⟩
def transferEvent : Nat := 86876
def frameStart : Nat := 86787
def rule : BoundRule := .product (.predecessor 0 86874 .coefficient) (.predecessor 1 86875 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86874 .coefficient)
      LeftBound86872.bound (LeftBound86872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86872.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86875 .coefficient)
      LeftAuthority86831.bound (LeftAuthority86831.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86832RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86831.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86831.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound86872.bound LeftAuthority86831.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86872.bound, LeftAuthority86831.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound86872.actual selector witness) * (LeftAuthority86831.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86876

namespace LeftBound86887
def owner : Owner := ⟨.program ⟨214⟩, ⟨15424⟩⟩
def transferEvent : Nat := 86887
def frameStart : Nat := 86787
def rule : BoundRule := .product (.predecessor 0 86885 .coefficient) (.predecessor 1 86886 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86885 .coefficient)
      LeftAuthority86842.bound (LeftAuthority86842.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86843RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86842.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86842.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86886 .coefficient)
      LeftAuthority86883.bound (LeftAuthority86883.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86883.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86883.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority86842.bound LeftAuthority86883.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86842.bound, LeftAuthority86883.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority86842.actual selector witness) * (LeftAuthority86883.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86887

namespace LeftBound86895
def owner : Owner := ⟨.program ⟨214⟩, ⟨15425⟩⟩
def transferEvent : Nat := 86895
def frameStart : Nat := 86787
def rule : BoundRule := .sum [.predecessor 0 86893 .coefficient, .predecessor 1 86894 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86893 .coefficient)
      LeftAuthority86891.bound (LeftAuthority86891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86891.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86891.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86894 .coefficient)
      LeftBound86887.bound (LeftBound86887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86889RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86887.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86887.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority86891.bound, LeftBound86887.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86891.bound, LeftBound86887.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority86891.actual selector witness, LeftBound86887.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86895

namespace LeftBound86899
def owner : Owner := ⟨.program ⟨214⟩, ⟨25300⟩⟩
def transferEvent : Nat := 86899
def frameStart : Nat := 86787
def rule : BoundRule := .sum [.predecessor 0 86897 .coefficient, .predecessor 1 86898 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86897 .coefficient)
      LeftBound86895.bound (LeftBound86895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86895.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86895.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86898 .coefficient)
      LeftBound86876.bound (LeftBound86876.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86881RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86876.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86876.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86895.bound, LeftBound86876.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86895.bound, LeftBound86876.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86895.actual selector witness, LeftBound86876.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86899

namespace LeftBound86912
def owner : Owner := ⟨.program ⟨214⟩, ⟨25298⟩⟩
def transferEvent : Nat := 86912
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 86910 .coefficient, .predecessor 1 86911 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86910 .coefficient)
      LeftBound86735.bound (LeftBound86735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86735.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86735.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86911 .coefficient)
      LeftBound86718.bound (LeftBound86718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events338.exact86725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86718.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86718.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86735.bound, LeftBound86718.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86735.bound, LeftBound86718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86735.actual selector witness, LeftBound86718.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86912

namespace LeftBound86915
def owner : Owner := ⟨.program ⟨214⟩, ⟨25298⟩⟩
def transferEvent : Nat := 86915
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 86909 .summary, .result 86725 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86909 .summary)
      LeftBound86737.bound (LeftBound86737.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19243⟩⟩) (rawTerms := some (Proof.Events339.exact86909RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound86737.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86725 .summary)
      LeftBound86720.bound (LeftBound86720.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25297⟩⟩) (rawTerms := some (Proof.Events338.exact86725RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound86720.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86737.bound, LeftBound86720.bound]
def bound : CoeffClass := .finite ⟨352024077676544, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86737.bound, LeftBound86720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86737.actual selector witness, LeftBound86720.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86915

namespace LeftBound86919
def owner : Owner := ⟨.program ⟨214⟩, ⟨27000⟩⟩
def transferEvent : Nat := 86919
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 86917 .coefficient) (.predecessor 1 86918 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86917 .coefficient)
      LeftBound86912.bound (LeftBound86912.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86916RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86912.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86912.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86918 .coefficient)
      LeftAuthority86640.bound (LeftAuthority86640.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events338.exact86641RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86640.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86640.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound86912.bound LeftAuthority86640.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86912.bound, LeftAuthority86640.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound86912.actual selector witness) * (LeftAuthority86640.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86919

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
