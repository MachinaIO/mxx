import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard191

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound28847
def owner : Owner := ⟨.program ⟨214⟩, ⟨11085⟩⟩
def transferEvent : Nat := 28847
def frameStart : Nat := 28797
def rule : BoundRule := .sum [.predecessor 0 28845 .coefficient, .predecessor 1 28846 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28845 .coefficient)
      LeftBound28830.bound (LeftBound28830.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound28830.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28846 .coefficient)
      LeftAuthority28843.bound (LeftAuthority28843.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority28843.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28830.bound, LeftAuthority28843.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28830.bound, LeftAuthority28843.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28830.actual selector witness, LeftAuthority28843.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28847

namespace LeftBound28850
def owner : Owner := ⟨.program ⟨214⟩, ⟨11086⟩⟩
def transferEvent : Nat := 28850
def frameStart : Nat := 28797
def rule : BoundRule := .identity (.predecessor 0 28849 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28849 .coefficient)
      LeftBound28847.bound (LeftBound28847.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound28847.derived selector witness)

def rawBound : CoeffClass := LeftBound28847.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28847.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound28847.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound28850

namespace LeftBound28856
def owner : Owner := ⟨.program ⟨214⟩, ⟨11087⟩⟩
def transferEvent : Nat := 28856
def frameStart : Nat := 28797
def rule : BoundRule := .product (.predecessor 0 28854 .coefficient) (.predecessor 1 28855 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28854 .coefficient)
      LeftAuthority28852.bound (LeftAuthority28852.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28853RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28852.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28852.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28855 .coefficient)
      LeftBound28850.bound (LeftBound28850.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28851RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28850.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28850.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority28852.bound LeftBound28850.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28852.bound, LeftBound28850.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority28852.actual selector witness) * (LeftBound28850.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28856

namespace LeftBound28872
def owner : Owner := ⟨.program ⟨214⟩, ⟨7838⟩⟩
def transferEvent : Nat := 28872
def frameStart : Nat := 28797
def rule : BoundRule := .scale (.predecessor 0 28870 .coefficient) (.value (.predecessor 1 28871 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28870 .coefficient)
      LeftAuthority28868.bound (LeftAuthority28868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28868.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28871 .coefficient)
      LeftAuthority28859.bound (LeftAuthority28859.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority28859.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority28868.bound LeftAuthority28859.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28868.bound, LeftAuthority28859.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority28868.actual selector witness) * (LeftAuthority28859.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound28872

namespace LeftBound28875
def owner : Owner := ⟨.program ⟨214⟩, ⟨6791⟩⟩
def transferEvent : Nat := 28875
def frameStart : Nat := 28797
def rule : BoundRule := .identity (.predecessor 0 28874 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28874 .coefficient)
      LeftAuthority28862.bound (LeftAuthority28862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28863RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28862.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28862.derived selector witness)

def rawBound : CoeffClass := LeftAuthority28862.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority28862.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound28875

namespace LeftBound28879
def owner : Owner := ⟨.program ⟨214⟩, ⟨7839⟩⟩
def transferEvent : Nat := 28879
def frameStart : Nat := 28797
def rule : BoundRule := .product (.predecessor 0 28877 .coefficient) (.predecessor 1 28878 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28877 .coefficient)
      LeftBound28875.bound (LeftBound28875.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28875.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28875.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28878 .coefficient)
      LeftBound28872.bound (LeftBound28872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28872.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound28875.bound LeftBound28872.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28875.bound, LeftBound28872.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound28875.actual selector witness) * (LeftBound28872.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28879

namespace LeftBound28884
def owner : Owner := ⟨.program ⟨214⟩, ⟨11088⟩⟩
def transferEvent : Nat := 28884
def frameStart : Nat := 28797
def rule : BoundRule := .sum [.predecessor 0 28882 .coefficient, .predecessor 1 28883 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28882 .coefficient)
      LeftBound28879.bound (LeftBound28879.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28881RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28879.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28879.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28883 .coefficient)
      LeftBound28856.bound (LeftBound28856.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28858RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28856.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28856.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28879.bound, LeftBound28856.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28879.bound, LeftBound28856.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28879.actual selector witness, LeftBound28856.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28884

namespace LeftBound28888
def owner : Owner := ⟨.program ⟨214⟩, ⟨25083⟩⟩
def transferEvent : Nat := 28888
def frameStart : Nat := 28797
def rule : BoundRule := .product (.predecessor 0 28886 .coefficient) (.predecessor 1 28887 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28886 .coefficient)
      LeftBound28884.bound (LeftBound28884.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28885RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28884.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28884.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28887 .coefficient)
      LeftAuthority28841.bound (LeftAuthority28841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28841.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28841.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound28884.bound LeftAuthority28841.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28884.bound, LeftAuthority28841.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound28884.actual selector witness) * (LeftAuthority28841.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28888

namespace LeftBound28899
def owner : Owner := ⟨.program ⟨214⟩, ⟨15128⟩⟩
def transferEvent : Nat := 28899
def frameStart : Nat := 28797
def rule : BoundRule := .product (.predecessor 0 28897 .coefficient) (.predecessor 1 28898 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28897 .coefficient)
      LeftAuthority28852.bound (LeftAuthority28852.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28853RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28852.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28852.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28898 .coefficient)
      LeftAuthority28895.bound (LeftAuthority28895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28895.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28895.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority28852.bound LeftAuthority28895.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28852.bound, LeftAuthority28895.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority28852.actual selector witness) * (LeftAuthority28895.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28899

namespace LeftBound28907
def owner : Owner := ⟨.program ⟨214⟩, ⟨15129⟩⟩
def transferEvent : Nat := 28907
def frameStart : Nat := 28797
def rule : BoundRule := .sum [.predecessor 0 28905 .coefficient, .predecessor 1 28906 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28905 .coefficient)
      LeftAuthority28903.bound (LeftAuthority28903.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28904RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28903.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28903.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28906 .coefficient)
      LeftBound28899.bound (LeftBound28899.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28901RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28899.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28899.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority28903.bound, LeftBound28899.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28903.bound, LeftBound28899.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority28903.actual selector witness, LeftBound28899.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28907

namespace LeftBound28911
def owner : Owner := ⟨.program ⟨214⟩, ⟨25084⟩⟩
def transferEvent : Nat := 28911
def frameStart : Nat := 28797
def rule : BoundRule := .sum [.predecessor 0 28909 .coefficient, .predecessor 1 28910 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28909 .coefficient)
      LeftBound28907.bound (LeftBound28907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28907.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28910 .coefficient)
      LeftBound28888.bound (LeftBound28888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28888.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28888.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28907.bound, LeftBound28888.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28907.bound, LeftBound28888.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28907.actual selector witness, LeftBound28888.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28911

namespace LeftBound28924
def owner : Owner := ⟨.program ⟨214⟩, ⟨25082⟩⟩
def transferEvent : Nat := 28924
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 28922 .coefficient, .predecessor 1 28923 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28922 .coefficient)
      LeftBound28745.bound (LeftBound28745.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28745.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28745.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28923 .coefficient)
      LeftBound28728.bound (LeftBound28728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28735RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28728.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28728.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28745.bound, LeftBound28728.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28745.bound, LeftBound28728.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28745.actual selector witness, LeftBound28728.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28924

namespace LeftBound28927
def owner : Owner := ⟨.program ⟨214⟩, ⟨25082⟩⟩
def transferEvent : Nat := 28927
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 28921 .summary, .result 28735 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28921 .summary)
      LeftBound28747.bound (LeftBound28747.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19183⟩⟩) (rawTerms := some (Proof.Events112.exact28921RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28747.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28735 .summary)
      LeftBound28730.bound (LeftBound28730.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25081⟩⟩) (rawTerms := some (Proof.Events112.exact28735RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28730.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28747.bound, LeftBound28730.bound]
def bound : CoeffClass := .finite ⟨352017970769920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28747.bound, LeftBound28730.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28747.actual selector witness, LeftBound28730.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28927

namespace LeftBound28931
def owner : Owner := ⟨.program ⟨214⟩, ⟨26822⟩⟩
def transferEvent : Nat := 28931
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 28929 .coefficient) (.predecessor 1 28930 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28929 .coefficient)
      LeftBound28924.bound (LeftBound28924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact28928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28924.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28924.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28930 .coefficient)
      LeftAuthority28650.bound (LeftAuthority28650.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28651RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28650.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28650.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound28924.bound LeftAuthority28650.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28924.bound, LeftAuthority28650.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound28924.actual selector witness) * (LeftAuthority28650.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28931

namespace LeftBound28932
def owner : Owner := ⟨.program ⟨214⟩, ⟨26822⟩⟩
def transferEvent : Nat := 28932
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26820⟩⟩]⟩ [⟨.result 28651 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28651 .coefficient)
      LeftAuthority28650.bound (LeftAuthority28650.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26820⟩⟩) (rawTerms := some (Proof.Events111.exact28651RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28650.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28650.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority28650.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28650.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority28650.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound28932

namespace LeftBound28933
def owner : Owner := ⟨.program ⟨214⟩, ⟨26822⟩⟩
def transferEvent : Nat := 28933
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 28928 .summary) (.transfer 28932) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28928 .summary)
      LeftBound28927.bound (LeftBound28927.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25082⟩⟩) (rawTerms := some (Proof.Events113.exact28928RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28927.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 28932)
      LeftBound28932.bound (LeftBound28932.actual selector witness) := by
  exact .transfer (LeftBound28932.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound28927.bound LeftBound28932.bound
def bound : CoeffClass := .finite ⟨1291911585013138718720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28927.bound, LeftBound28932.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound28927.actual selector witness) * (LeftBound28932.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28933

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
