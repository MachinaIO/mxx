import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard459

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound67885
def owner : Owner := ⟨.program ⟨214⟩, ⟨12364⟩⟩
def transferEvent : Nat := 67885
def frameStart : Nat := 67852
def rule : BoundRule := .identity (.predecessor 0 67884 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67884 .coefficient)
      LeftBound67881.bound (LeftBound67881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67883RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67881.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67881.derived selector witness)

def rawBound : CoeffClass := LeftBound67881.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound67881.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound67885

namespace LeftBound67902
def owner : Owner := ⟨.program ⟨214⟩, ⟨12462⟩⟩
def transferEvent : Nat := 67902
def frameStart : Nat := 67852
def rule : BoundRule := .sum [.predecessor 0 67900 .coefficient, .predecessor 1 67901 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67900 .coefficient)
      LeftBound67885.bound (LeftBound67885.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound67885.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67901 .coefficient)
      LeftAuthority67898.bound (LeftAuthority67898.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority67898.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67885.bound, LeftAuthority67898.bound]
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67885.bound, LeftAuthority67898.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67885.actual selector witness, LeftAuthority67898.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67902

namespace LeftBound67905
def owner : Owner := ⟨.program ⟨214⟩, ⟨12463⟩⟩
def transferEvent : Nat := 67905
def frameStart : Nat := 67852
def rule : BoundRule := .identity (.predecessor 0 67904 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67904 .coefficient)
      LeftBound67902.bound (LeftBound67902.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound67902.derived selector witness)

def rawBound : CoeffClass := LeftBound67902.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound67902.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound67905

namespace LeftBound67911
def owner : Owner := ⟨.program ⟨214⟩, ⟨12464⟩⟩
def transferEvent : Nat := 67911
def frameStart : Nat := 67852
def rule : BoundRule := .product (.predecessor 0 67909 .coefficient) (.predecessor 1 67910 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67909 .coefficient)
      LeftAuthority67907.bound (LeftAuthority67907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67907.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67907.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67910 .coefficient)
      LeftBound67905.bound (LeftBound67905.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67906RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67905.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67905.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority67907.bound LeftBound67905.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67907.bound, LeftBound67905.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority67907.actual selector witness) * (LeftBound67905.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67911

namespace LeftBound67927
def owner : Owner := ⟨.program ⟨214⟩, ⟨7868⟩⟩
def transferEvent : Nat := 67927
def frameStart : Nat := 67852
def rule : BoundRule := .scale (.predecessor 0 67925 .coefficient) (.value (.predecessor 1 67926 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67925 .coefficient)
      LeftAuthority67923.bound (LeftAuthority67923.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67923.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67923.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67926 .coefficient)
      LeftAuthority67914.bound (LeftAuthority67914.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority67914.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority67923.bound LeftAuthority67914.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67923.bound, LeftAuthority67914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority67923.actual selector witness) * (LeftAuthority67914.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound67927

namespace LeftBound67930
def owner : Owner := ⟨.program ⟨214⟩, ⟨6765⟩⟩
def transferEvent : Nat := 67930
def frameStart : Nat := 67852
def rule : BoundRule := .identity (.predecessor 0 67929 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67929 .coefficient)
      LeftAuthority67917.bound (LeftAuthority67917.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67917.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67917.derived selector witness)

def rawBound : CoeffClass := LeftAuthority67917.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67917.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority67917.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound67930

namespace LeftBound67934
def owner : Owner := ⟨.program ⟨214⟩, ⟨7869⟩⟩
def transferEvent : Nat := 67934
def frameStart : Nat := 67852
def rule : BoundRule := .product (.predecessor 0 67932 .coefficient) (.predecessor 1 67933 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67932 .coefficient)
      LeftBound67930.bound (LeftBound67930.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67931RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67930.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67930.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67933 .coefficient)
      LeftBound67927.bound (LeftBound67927.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67927.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67927.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound67930.bound LeftBound67927.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67930.bound, LeftBound67927.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound67930.actual selector witness) * (LeftBound67927.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67934

namespace LeftBound67939
def owner : Owner := ⟨.program ⟨214⟩, ⟨12465⟩⟩
def transferEvent : Nat := 67939
def frameStart : Nat := 67852
def rule : BoundRule := .sum [.predecessor 0 67937 .coefficient, .predecessor 1 67938 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67937 .coefficient)
      LeftBound67934.bound (LeftBound67934.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67936RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67934.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67934.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67938 .coefficient)
      LeftBound67911.bound (LeftBound67911.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67913RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67911.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67911.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67934.bound, LeftBound67911.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67934.bound, LeftBound67911.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67934.actual selector witness, LeftBound67911.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67939

namespace LeftBound67943
def owner : Owner := ⟨.program ⟨214⟩, ⟨25371⟩⟩
def transferEvent : Nat := 67943
def frameStart : Nat := 67852
def rule : BoundRule := .product (.predecessor 0 67941 .coefficient) (.predecessor 1 67942 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67941 .coefficient)
      LeftBound67939.bound (LeftBound67939.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67939.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67939.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67942 .coefficient)
      LeftAuthority67896.bound (LeftAuthority67896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67897RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67896.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67896.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound67939.bound LeftAuthority67896.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67939.bound, LeftAuthority67896.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound67939.actual selector witness) * (LeftAuthority67896.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67943

namespace LeftBound67954
def owner : Owner := ⟨.program ⟨214⟩, ⟨16463⟩⟩
def transferEvent : Nat := 67954
def frameStart : Nat := 67852
def rule : BoundRule := .product (.predecessor 0 67952 .coefficient) (.predecessor 1 67953 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67952 .coefficient)
      LeftAuthority67907.bound (LeftAuthority67907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67907.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67907.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67953 .coefficient)
      LeftAuthority67950.bound (LeftAuthority67950.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67951RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67950.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67950.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority67907.bound LeftAuthority67950.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67907.bound, LeftAuthority67950.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority67907.actual selector witness) * (LeftAuthority67950.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67954

namespace LeftBound67962
def owner : Owner := ⟨.program ⟨214⟩, ⟨16464⟩⟩
def transferEvent : Nat := 67962
def frameStart : Nat := 67852
def rule : BoundRule := .sum [.predecessor 0 67960 .coefficient, .predecessor 1 67961 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67960 .coefficient)
      LeftAuthority67958.bound (LeftAuthority67958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67959RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67958.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67958.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67961 .coefficient)
      LeftBound67954.bound (LeftBound67954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67956RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67954.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67954.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority67958.bound, LeftBound67954.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67958.bound, LeftBound67954.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority67958.actual selector witness, LeftBound67954.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67962

namespace LeftBound67966
def owner : Owner := ⟨.program ⟨214⟩, ⟨25372⟩⟩
def transferEvent : Nat := 67966
def frameStart : Nat := 67852
def rule : BoundRule := .sum [.predecessor 0 67964 .coefficient, .predecessor 1 67965 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67964 .coefficient)
      LeftBound67962.bound (LeftBound67962.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67963RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67962.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67962.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67965 .coefficient)
      LeftBound67943.bound (LeftBound67943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67943.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67962.bound, LeftBound67943.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67962.bound, LeftBound67943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67962.actual selector witness, LeftBound67943.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67966

namespace LeftBound67979
def owner : Owner := ⟨.program ⟨214⟩, ⟨25370⟩⟩
def transferEvent : Nat := 67979
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 67977 .coefficient, .predecessor 1 67978 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67977 .coefficient)
      LeftBound67800.bound (LeftBound67800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67800.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67800.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67978 .coefficient)
      LeftBound67783.bound (LeftBound67783.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67783.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67783.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67800.bound, LeftBound67783.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67800.bound, LeftBound67783.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67800.actual selector witness, LeftBound67783.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67979

namespace LeftBound67982
def owner : Owner := ⟨.program ⟨214⟩, ⟨25370⟩⟩
def transferEvent : Nat := 67982
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 67976 .summary, .result 67790 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67976 .summary)
      LeftBound67802.bound (LeftBound67802.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19887⟩⟩) (rawTerms := some (Proof.Events265.exact67976RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67802.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67790 .summary)
      LeftBound67785.bound (LeftBound67785.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25369⟩⟩) (rawTerms := some (Proof.Events264.exact67790RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67785.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67802.bound, LeftBound67785.bound]
def bound : CoeffClass := .finite ⟨352127895089152, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67802.bound, LeftBound67785.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67802.actual selector witness, LeftBound67785.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67982

namespace LeftBound67986
def owner : Owner := ⟨.program ⟨214⟩, ⟨28940⟩⟩
def transferEvent : Nat := 67986
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 67984 .coefficient) (.predecessor 1 67985 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67984 .coefficient)
      LeftBound67979.bound (LeftBound67979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67979.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67985 .coefficient)
      LeftAuthority67705.bound (LeftAuthority67705.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67706RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67705.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67705.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound67979.bound LeftAuthority67705.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67979.bound, LeftAuthority67705.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound67979.actual selector witness) * (LeftAuthority67705.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67986

namespace LeftBound67987
def owner : Owner := ⟨.program ⟨214⟩, ⟨28940⟩⟩
def transferEvent : Nat := 67987
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28938⟩⟩]⟩ [⟨.result 67706 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67706 .coefficient)
      LeftAuthority67705.bound (LeftAuthority67705.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28938⟩⟩) (rawTerms := some (Proof.Events264.exact67706RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67705.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67705.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority67705.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67705.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority67705.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound67987

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
