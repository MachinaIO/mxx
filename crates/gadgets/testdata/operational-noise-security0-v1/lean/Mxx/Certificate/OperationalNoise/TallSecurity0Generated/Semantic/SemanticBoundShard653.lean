import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard652

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound95873
def owner : Owner := ⟨.program ⟨214⟩, ⟨6767⟩⟩
def transferEvent : Nat := 95873
def frameStart : Nat := 95807
def rule : BoundRule := .identity (.predecessor 0 95872 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95872 .coefficient)
      LeftAuthority95860.bound (LeftAuthority95860.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95861RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95860.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95860.derived selector witness)

def rawBound : CoeffClass := LeftAuthority95860.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95860.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority95860.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound95873

namespace LeftBound95877
def owner : Owner := ⟨.program ⟨214⟩, ⟨7875⟩⟩
def transferEvent : Nat := 95877
def frameStart : Nat := 95807
def rule : BoundRule := .product (.predecessor 0 95875 .coefficient) (.predecessor 1 95876 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95875 .coefficient)
      LeftBound95873.bound (LeftBound95873.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95874RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95873.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95873.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95876 .coefficient)
      LeftBound95870.bound (LeftBound95870.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95871RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95870.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95870.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95873.bound LeftBound95870.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95873.bound, LeftBound95870.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95873.actual selector witness) * (LeftBound95870.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95877

namespace LeftBound95882
def owner : Owner := ⟨.program ⟨214⟩, ⟨12853⟩⟩
def transferEvent : Nat := 95882
def frameStart : Nat := 95807
def rule : BoundRule := .sum [.predecessor 0 95880 .coefficient, .predecessor 1 95881 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95880 .coefficient)
      LeftBound95877.bound (LeftBound95877.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95879RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95877.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95877.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95881 .coefficient)
      LeftBound95854.bound (LeftBound95854.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95854.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95854.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95877.bound, LeftBound95854.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95877.bound, LeftBound95854.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95877.actual selector witness, LeftBound95854.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95882

namespace LeftBound95886
def owner : Owner := ⟨.program ⟨214⟩, ⟨25517⟩⟩
def transferEvent : Nat := 95886
def frameStart : Nat := 95807
def rule : BoundRule := .product (.predecessor 0 95884 .coefficient) (.predecessor 1 95885 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95884 .coefficient)
      LeftBound95882.bound (LeftBound95882.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95883RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95882.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95882.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95885 .coefficient)
      LeftAuthority95839.bound (LeftAuthority95839.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95839.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95839.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95882.bound LeftAuthority95839.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95882.bound, LeftAuthority95839.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95882.actual selector witness) * (LeftAuthority95839.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95886

namespace LeftBound95897
def owner : Owner := ⟨.program ⟨214⟩, ⟨16625⟩⟩
def transferEvent : Nat := 95897
def frameStart : Nat := 95807
def rule : BoundRule := .product (.predecessor 0 95895 .coefficient) (.predecessor 1 95896 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95895 .coefficient)
      LeftAuthority95850.bound (LeftAuthority95850.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95851RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95850.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95850.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95896 .coefficient)
      LeftAuthority95893.bound (LeftAuthority95893.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95894RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95893.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95893.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority95850.bound LeftAuthority95893.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95850.bound, LeftAuthority95893.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority95850.actual selector witness) * (LeftAuthority95893.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95897

namespace LeftBound95905
def owner : Owner := ⟨.program ⟨214⟩, ⟨16626⟩⟩
def transferEvent : Nat := 95905
def frameStart : Nat := 95807
def rule : BoundRule := .sum [.predecessor 0 95903 .coefficient, .predecessor 1 95904 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95903 .coefficient)
      LeftAuthority95901.bound (LeftAuthority95901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95902RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95901.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95901.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95904 .coefficient)
      LeftBound95897.bound (LeftBound95897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95897.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95897.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority95901.bound, LeftBound95897.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95901.bound, LeftBound95897.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority95901.actual selector witness, LeftBound95897.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95905

namespace LeftBound95909
def owner : Owner := ⟨.program ⟨214⟩, ⟨25518⟩⟩
def transferEvent : Nat := 95909
def frameStart : Nat := 95807
def rule : BoundRule := .sum [.predecessor 0 95907 .coefficient, .predecessor 1 95908 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95907 .coefficient)
      LeftBound95905.bound (LeftBound95905.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95906RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95905.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95905.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95908 .coefficient)
      LeftBound95886.bound (LeftBound95886.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95886.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95886.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95905.bound, LeftBound95886.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95905.bound, LeftBound95886.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95905.actual selector witness, LeftBound95886.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95909

namespace LeftBound95922
def owner : Owner := ⟨.program ⟨214⟩, ⟨25516⟩⟩
def transferEvent : Nat := 95922
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 95920 .coefficient, .predecessor 1 95921 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95920 .coefficient)
      LeftBound95767.bound (LeftBound95767.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95919RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95767.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95767.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95921 .coefficient)
      LeftBound95750.bound (LeftBound95750.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95757RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95750.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95750.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95767.bound, LeftBound95750.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95767.bound, LeftBound95750.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95767.actual selector witness, LeftBound95750.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95922

namespace LeftBound95925
def owner : Owner := ⟨.program ⟨214⟩, ⟨25516⟩⟩
def transferEvent : Nat := 95925
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 95919 .summary, .result 95757 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95919 .summary)
      LeftBound95769.bound (LeftBound95769.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20024⟩⟩) (rawTerms := some (Proof.Events374.exact95919RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95769.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95757 .summary)
      LeftBound95752.bound (LeftBound95752.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25515⟩⟩) (rawTerms := some (Proof.Events374.exact95757RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95752.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95769.bound, LeftBound95752.bound]
def bound : CoeffClass := .finite ⟨352146215809024, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95769.bound, LeftBound95752.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95769.actual selector witness, LeftBound95752.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95925

namespace LeftBound95929
def owner : Owner := ⟨.program ⟨214⟩, ⟨29352⟩⟩
def transferEvent : Nat := 95929
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 95927 .coefficient) (.predecessor 1 95928 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95927 .coefficient)
      LeftBound95922.bound (LeftBound95922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95926RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95922.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95922.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95928 .coefficient)
      LeftAuthority95672.bound (LeftAuthority95672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95672.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95672.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95922.bound LeftAuthority95672.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95922.bound, LeftAuthority95672.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95922.actual selector witness) * (LeftAuthority95672.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95929

namespace LeftBound95930
def owner : Owner := ⟨.program ⟨214⟩, ⟨29352⟩⟩
def transferEvent : Nat := 95930
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29350⟩⟩]⟩ [⟨.result 95673 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95673 .coefficient)
      LeftAuthority95672.bound (LeftAuthority95672.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29350⟩⟩) (rawTerms := some (Proof.Events373.exact95673RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95672.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95672.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority95672.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95672.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority95672.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound95930

namespace LeftBound95931
def owner : Owner := ⟨.program ⟨214⟩, ⟨29352⟩⟩
def transferEvent : Nat := 95931
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 95926 .summary) (.transfer 95930) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95926 .summary)
      LeftBound95925.bound (LeftBound95925.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25516⟩⟩) (rawTerms := some (Proof.Events374.exact95926RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95925.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 95930)
      LeftBound95930.bound (LeftBound95930.actual selector witness) := by
  exact .transfer (LeftBound95930.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95925.bound LeftBound95930.bound
def bound : CoeffClass := .finite ⟨1292382246358571024384, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95925.bound, LeftBound95930.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95925.actual selector witness) * (LeftBound95930.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95931

namespace LeftBound95942
def owner : Owner := ⟨.program ⟨214⟩, ⟨22399⟩⟩
def transferEvent : Nat := 95942
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 95940 .coefficient) (.value (.predecessor 1 95941 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95940 .coefficient)
      LeftAuthority95938.bound (LeftAuthority95938.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95939RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95938.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95938.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95941 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority95938.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95938.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority95938.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound95942

namespace LeftBound95946
def owner : Owner := ⟨.program ⟨214⟩, ⟨22400⟩⟩
def transferEvent : Nat := 95946
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 95944 .coefficient) (.predecessor 1 95945 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95944 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95945 .coefficient)
      LeftBound95942.bound (LeftBound95942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95943RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95942.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95942.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound95942.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound95942.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound95942.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95946

namespace LeftBound95947
def owner : Owner := ⟨.program ⟨214⟩, ⟨22400⟩⟩
def transferEvent : Nat := 95947
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22397⟩⟩]⟩ [⟨.result 95939 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95939 .coefficient)
      LeftAuthority95938.bound (LeftAuthority95938.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22397⟩⟩) (rawTerms := some (Proof.Events374.exact95939RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95938.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95938.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority95938.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95938.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority95938.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound95947

namespace LeftBound95948
def owner : Owner := ⟨.program ⟨214⟩, ⟨22400⟩⟩
def transferEvent : Nat := 95948
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 95947) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 95947)
      LeftBound95947.bound (LeftBound95947.actual selector witness) := by
  exact .transfer (LeftBound95947.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound95947.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound95947.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound95947.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95948

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
