import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard249

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound37691
def owner : Owner := ⟨.program ⟨214⟩, ⟨12867⟩⟩
def transferEvent : Nat := 37691
def frameStart : Nat := 37638
def rule : BoundRule := .identity (.predecessor 0 37690 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37690 .coefficient)
      LeftBound37688.bound (LeftBound37688.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound37688.derived selector witness)

def rawBound : CoeffClass := LeftBound37688.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37688.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound37688.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound37691

namespace LeftBound37697
def owner : Owner := ⟨.program ⟨214⟩, ⟨12868⟩⟩
def transferEvent : Nat := 37697
def frameStart : Nat := 37638
def rule : BoundRule := .product (.predecessor 0 37695 .coefficient) (.predecessor 1 37696 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37695 .coefficient)
      LeftAuthority37693.bound (LeftAuthority37693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37694RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37693.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37693.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37696 .coefficient)
      LeftBound37691.bound (LeftBound37691.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37692RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37691.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37691.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority37693.bound LeftBound37691.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37693.bound, LeftBound37691.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority37693.actual selector witness) * (LeftBound37691.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37697

namespace LeftBound37713
def owner : Owner := ⟨.program ⟨214⟩, ⟨7874⟩⟩
def transferEvent : Nat := 37713
def frameStart : Nat := 37638
def rule : BoundRule := .scale (.predecessor 0 37711 .coefficient) (.value (.predecessor 1 37712 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37711 .coefficient)
      LeftAuthority37709.bound (LeftAuthority37709.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37710RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37709.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37709.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37712 .coefficient)
      LeftAuthority37700.bound (LeftAuthority37700.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority37700.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority37709.bound LeftAuthority37700.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37709.bound, LeftAuthority37700.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority37709.actual selector witness) * (LeftAuthority37700.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound37713

namespace LeftBound37716
def owner : Owner := ⟨.program ⟨214⟩, ⟨6767⟩⟩
def transferEvent : Nat := 37716
def frameStart : Nat := 37638
def rule : BoundRule := .identity (.predecessor 0 37715 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37715 .coefficient)
      LeftAuthority37703.bound (LeftAuthority37703.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37704RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37703.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37703.derived selector witness)

def rawBound : CoeffClass := LeftAuthority37703.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37703.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority37703.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound37716

namespace LeftBound37720
def owner : Owner := ⟨.program ⟨214⟩, ⟨7875⟩⟩
def transferEvent : Nat := 37720
def frameStart : Nat := 37638
def rule : BoundRule := .product (.predecessor 0 37718 .coefficient) (.predecessor 1 37719 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37718 .coefficient)
      LeftBound37716.bound (LeftBound37716.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37716.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37716.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37719 .coefficient)
      LeftBound37713.bound (LeftBound37713.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37714RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37713.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37713.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound37716.bound LeftBound37713.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37716.bound, LeftBound37713.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound37716.actual selector witness) * (LeftBound37713.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37720

namespace LeftBound37725
def owner : Owner := ⟨.program ⟨214⟩, ⟨12869⟩⟩
def transferEvent : Nat := 37725
def frameStart : Nat := 37638
def rule : BoundRule := .sum [.predecessor 0 37723 .coefficient, .predecessor 1 37724 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37723 .coefficient)
      LeftBound37720.bound (LeftBound37720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37722RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37720.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37720.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37724 .coefficient)
      LeftBound37697.bound (LeftBound37697.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37699RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37697.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37697.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37720.bound, LeftBound37697.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37720.bound, LeftBound37697.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound37720.actual selector witness, LeftBound37697.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37725

namespace LeftBound37729
def owner : Owner := ⟨.program ⟨214⟩, ⟨25540⟩⟩
def transferEvent : Nat := 37729
def frameStart : Nat := 37638
def rule : BoundRule := .product (.predecessor 0 37727 .coefficient) (.predecessor 1 37728 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37727 .coefficient)
      LeftBound37725.bound (LeftBound37725.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37726RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37725.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37725.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37728 .coefficient)
      LeftAuthority37682.bound (LeftAuthority37682.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37682.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound37725.bound LeftAuthority37682.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37725.bound, LeftAuthority37682.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound37725.actual selector witness) * (LeftAuthority37682.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37729

namespace LeftBound37740
def owner : Owner := ⟨.program ⟨214⟩, ⟨16643⟩⟩
def transferEvent : Nat := 37740
def frameStart : Nat := 37638
def rule : BoundRule := .product (.predecessor 0 37738 .coefficient) (.predecessor 1 37739 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37738 .coefficient)
      LeftAuthority37693.bound (LeftAuthority37693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37694RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37693.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37693.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37739 .coefficient)
      LeftAuthority37736.bound (LeftAuthority37736.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37737RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37736.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37736.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority37693.bound LeftAuthority37736.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37693.bound, LeftAuthority37736.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority37693.actual selector witness) * (LeftAuthority37736.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37740

namespace LeftBound37748
def owner : Owner := ⟨.program ⟨214⟩, ⟨16644⟩⟩
def transferEvent : Nat := 37748
def frameStart : Nat := 37638
def rule : BoundRule := .sum [.predecessor 0 37746 .coefficient, .predecessor 1 37747 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37746 .coefficient)
      LeftAuthority37744.bound (LeftAuthority37744.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37744.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37744.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37747 .coefficient)
      LeftBound37740.bound (LeftBound37740.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37742RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37740.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37740.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority37744.bound, LeftBound37740.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37744.bound, LeftBound37740.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority37744.actual selector witness, LeftBound37740.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37748

namespace LeftBound37752
def owner : Owner := ⟨.program ⟨214⟩, ⟨25541⟩⟩
def transferEvent : Nat := 37752
def frameStart : Nat := 37638
def rule : BoundRule := .sum [.predecessor 0 37750 .coefficient, .predecessor 1 37751 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37750 .coefficient)
      LeftBound37748.bound (LeftBound37748.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37749RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37748.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37748.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37751 .coefficient)
      LeftBound37729.bound (LeftBound37729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37734RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37729.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37729.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37748.bound, LeftBound37729.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37748.bound, LeftBound37729.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound37748.actual selector witness, LeftBound37729.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37752

namespace LeftBound37765
def owner : Owner := ⟨.program ⟨214⟩, ⟨25539⟩⟩
def transferEvent : Nat := 37765
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 37763 .coefficient, .predecessor 1 37764 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37763 .coefficient)
      LeftBound37586.bound (LeftBound37586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37586.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37764 .coefficient)
      LeftBound37569.bound (LeftBound37569.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37569.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37569.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37586.bound, LeftBound37569.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37586.bound, LeftBound37569.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound37586.actual selector witness, LeftBound37569.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37765

namespace LeftBound37768
def owner : Owner := ⟨.program ⟨214⟩, ⟨25539⟩⟩
def transferEvent : Nat := 37768
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 37762 .summary, .result 37576 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37762 .summary)
      LeftBound37588.bound (LeftBound37588.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20043⟩⟩) (rawTerms := some (Proof.Events147.exact37762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound37588.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37576 .summary)
      LeftBound37571.bound (LeftBound37571.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25538⟩⟩) (rawTerms := some (Proof.Events146.exact37576RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound37571.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37588.bound, LeftBound37571.bound]
def bound : CoeffClass := .finite ⟨352146215809024, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37588.bound, LeftBound37571.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound37588.actual selector witness, LeftBound37571.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37768

namespace LeftBound37772
def owner : Owner := ⟨.program ⟨214⟩, ⟨29413⟩⟩
def transferEvent : Nat := 37772
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 37770 .coefficient) (.predecessor 1 37771 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37770 .coefficient)
      LeftBound37765.bound (LeftBound37765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37769RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37765.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37765.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37771 .coefficient)
      LeftAuthority37491.bound (LeftAuthority37491.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37492RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37491.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37491.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound37765.bound LeftAuthority37491.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37765.bound, LeftAuthority37491.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound37765.actual selector witness) * (LeftAuthority37491.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37772

namespace LeftBound37773
def owner : Owner := ⟨.program ⟨214⟩, ⟨29413⟩⟩
def transferEvent : Nat := 37773
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29411⟩⟩]⟩ [⟨.result 37492 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37492 .coefficient)
      LeftAuthority37491.bound (LeftAuthority37491.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29411⟩⟩) (rawTerms := some (Proof.Events146.exact37492RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37491.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37491.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority37491.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37491.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority37491.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound37773

namespace LeftBound37774
def owner : Owner := ⟨.program ⟨214⟩, ⟨29413⟩⟩
def transferEvent : Nat := 37774
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 37769 .summary) (.transfer 37773) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37769 .summary)
      LeftBound37768.bound (LeftBound37768.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25539⟩⟩) (rawTerms := some (Proof.Events147.exact37769RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound37768.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 37773)
      LeftBound37773.bound (LeftBound37773.actual selector witness) := by
  exact .transfer (LeftBound37773.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound37768.bound LeftBound37773.bound
def bound : CoeffClass := .finite ⟨1292382246358571024384, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37768.bound, LeftBound37773.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound37768.actual selector witness) * (LeftBound37773.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37774

namespace LeftBound37785
def owner : Owner := ⟨.program ⟨214⟩, ⟨22418⟩⟩
def transferEvent : Nat := 37785
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 37783 .coefficient) (.value (.predecessor 1 37784 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37783 .coefficient)
      LeftAuthority37781.bound (LeftAuthority37781.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37782RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37781.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37781.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37784 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority37781.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37781.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority37781.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound37785

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
