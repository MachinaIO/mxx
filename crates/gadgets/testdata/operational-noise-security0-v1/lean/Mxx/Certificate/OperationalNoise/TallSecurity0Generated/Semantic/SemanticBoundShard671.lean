import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard670

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound98024
def owner : Owner := ⟨.program ⟨214⟩, ⟨14742⟩⟩
def transferEvent : Nat := 98024
def frameStart : Nat := 97977
def rule : BoundRule := .product (.predecessor 0 98022 .coefficient) (.predecessor 1 98023 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98022 .coefficient)
      LeftAuthority98020.bound (LeftAuthority98020.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact98021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98020.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98020.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98023 .coefficient)
      LeftBound98018.bound (LeftBound98018.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact98019RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98018.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98018.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority98020.bound LeftBound98018.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98020.bound, LeftBound98018.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority98020.actual selector witness) * (LeftBound98018.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98024

namespace LeftBound98040
def owner : Owner := ⟨.program ⟨214⟩, ⟨7859⟩⟩
def transferEvent : Nat := 98040
def frameStart : Nat := 97977
def rule : BoundRule := .scale (.predecessor 0 98038 .coefficient) (.value (.predecessor 1 98039 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98038 .coefficient)
      LeftAuthority98036.bound (LeftAuthority98036.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact98037RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98036.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98036.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98039 .coefficient)
      LeftAuthority98027.bound (LeftAuthority98027.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority98027.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority98036.bound LeftAuthority98027.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98036.bound, LeftAuthority98027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority98036.actual selector witness) * (LeftAuthority98027.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound98040

namespace LeftBound98043
def owner : Owner := ⟨.program ⟨214⟩, ⟨6762⟩⟩
def transferEvent : Nat := 98043
def frameStart : Nat := 97977
def rule : BoundRule := .identity (.predecessor 0 98042 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98042 .coefficient)
      LeftAuthority98030.bound (LeftAuthority98030.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact98031RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98030.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98030.derived selector witness)

def rawBound : CoeffClass := LeftAuthority98030.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98030.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority98030.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound98043

namespace LeftBound98047
def owner : Owner := ⟨.program ⟨214⟩, ⟨7860⟩⟩
def transferEvent : Nat := 98047
def frameStart : Nat := 97977
def rule : BoundRule := .product (.predecessor 0 98045 .coefficient) (.predecessor 1 98046 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98045 .coefficient)
      LeftBound98043.bound (LeftBound98043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact98044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98043.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98046 .coefficient)
      LeftBound98040.bound (LeftBound98040.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact98041RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98040.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98040.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98043.bound LeftBound98040.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98043.bound, LeftBound98040.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98043.actual selector witness) * (LeftBound98040.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98047

namespace LeftBound98052
def owner : Owner := ⟨.program ⟨214⟩, ⟨14743⟩⟩
def transferEvent : Nat := 98052
def frameStart : Nat := 97977
def rule : BoundRule := .sum [.predecessor 0 98050 .coefficient, .predecessor 1 98051 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98050 .coefficient)
      LeftBound98047.bound (LeftBound98047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98047.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98047.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98051 .coefficient)
      LeftBound98024.bound (LeftBound98024.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact98026RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98024.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98024.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98047.bound, LeftBound98024.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98047.bound, LeftBound98024.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98047.actual selector witness, LeftBound98024.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98052

namespace LeftBound98056
def owner : Owner := ⟨.program ⟨214⟩, ⟨26210⟩⟩
def transferEvent : Nat := 98056
def frameStart : Nat := 97977
def rule : BoundRule := .product (.predecessor 0 98054 .coefficient) (.predecessor 1 98055 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98054 .coefficient)
      LeftBound98052.bound (LeftBound98052.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98053RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98052.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98052.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98055 .coefficient)
      LeftAuthority98009.bound (LeftAuthority98009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact98010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98009.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98009.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98052.bound LeftAuthority98009.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98052.bound, LeftAuthority98009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98052.actual selector witness) * (LeftAuthority98009.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98056

namespace LeftBound98067
def owner : Owner := ⟨.program ⟨214⟩, ⟨16170⟩⟩
def transferEvent : Nat := 98067
def frameStart : Nat := 97977
def rule : BoundRule := .product (.predecessor 0 98065 .coefficient) (.predecessor 1 98066 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98065 .coefficient)
      LeftAuthority98020.bound (LeftAuthority98020.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact98021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98020.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98020.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98066 .coefficient)
      LeftAuthority98063.bound (LeftAuthority98063.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98063.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98063.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority98020.bound LeftAuthority98063.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98020.bound, LeftAuthority98063.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority98020.actual selector witness) * (LeftAuthority98063.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98067

namespace LeftBound98075
def owner : Owner := ⟨.program ⟨214⟩, ⟨16171⟩⟩
def transferEvent : Nat := 98075
def frameStart : Nat := 97977
def rule : BoundRule := .sum [.predecessor 0 98073 .coefficient, .predecessor 1 98074 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98073 .coefficient)
      LeftAuthority98071.bound (LeftAuthority98071.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98071.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98071.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98074 .coefficient)
      LeftBound98067.bound (LeftBound98067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98069RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98067.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority98071.bound, LeftBound98067.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98071.bound, LeftBound98067.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority98071.actual selector witness, LeftBound98067.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98075

namespace LeftBound98079
def owner : Owner := ⟨.program ⟨214⟩, ⟨26211⟩⟩
def transferEvent : Nat := 98079
def frameStart : Nat := 97977
def rule : BoundRule := .sum [.predecessor 0 98077 .coefficient, .predecessor 1 98078 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98077 .coefficient)
      LeftBound98075.bound (LeftBound98075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98075.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98075.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98078 .coefficient)
      LeftBound98056.bound (LeftBound98056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98061RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98056.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98056.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98075.bound, LeftBound98056.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98075.bound, LeftBound98056.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98075.actual selector witness, LeftBound98056.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98079

namespace LeftBound98092
def owner : Owner := ⟨.program ⟨214⟩, ⟨26209⟩⟩
def transferEvent : Nat := 98092
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 98090 .coefficient, .predecessor 1 98091 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98090 .coefficient)
      LeftBound97937.bound (LeftBound97937.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98089RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97937.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97937.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98091 .coefficient)
      LeftBound97920.bound (LeftBound97920.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact97927RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97920.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97920.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97937.bound, LeftBound97920.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97937.bound, LeftBound97920.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97937.actual selector witness, LeftBound97920.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98092

namespace LeftBound98095
def owner : Owner := ⟨.program ⟨214⟩, ⟨26209⟩⟩
def transferEvent : Nat := 98095
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 98089 .summary, .result 97927 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98089 .summary)
      LeftBound97939.bound (LeftBound97939.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19664⟩⟩) (rawTerms := some (Proof.Events383.exact98089RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97939.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97927 .summary)
      LeftBound97922.bound (LeftBound97922.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26208⟩⟩) (rawTerms := some (Proof.Events382.exact97927RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97922.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97939.bound, LeftBound97922.bound]
def bound : CoeffClass := .finite ⟨352091253649408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97939.bound, LeftBound97922.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97939.actual selector witness, LeftBound97922.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98095

namespace LeftBound98099
def owner : Owner := ⟨.program ⟨214⟩, ⟨28267⟩⟩
def transferEvent : Nat := 98099
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 98097 .coefficient) (.predecessor 1 98098 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98097 .coefficient)
      LeftBound98092.bound (LeftBound98092.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98092.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98092.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98098 .coefficient)
      LeftAuthority97842.bound (LeftAuthority97842.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact97843RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97842.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97842.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98092.bound LeftAuthority97842.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98092.bound, LeftAuthority97842.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98092.actual selector witness) * (LeftAuthority97842.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98099

namespace LeftBound98100
def owner : Owner := ⟨.program ⟨214⟩, ⟨28267⟩⟩
def transferEvent : Nat := 98100
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28265⟩⟩]⟩ [⟨.result 97843 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97843 .coefficient)
      LeftAuthority97842.bound (LeftAuthority97842.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28265⟩⟩) (rawTerms := some (Proof.Events382.exact97843RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97842.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97842.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority97842.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97842.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority97842.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound98100

namespace LeftBound98101
def owner : Owner := ⟨.program ⟨214⟩, ⟨28267⟩⟩
def transferEvent : Nat := 98101
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 98096 .summary) (.transfer 98100) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98096 .summary)
      LeftBound98095.bound (LeftBound98095.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26209⟩⟩) (rawTerms := some (Proof.Events383.exact98096RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98095.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 98100)
      LeftBound98100.bound (LeftBound98100.actual selector witness) := by
  exact .transfer (LeftBound98100.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98095.bound LeftBound98100.bound
def bound : CoeffClass := .finite ⟨1292180534353385750528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98095.bound, LeftBound98100.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98095.actual selector witness) * (LeftBound98100.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98101

namespace LeftBound98112
def owner : Owner := ⟨.program ⟨214⟩, ⟨21679⟩⟩
def transferEvent : Nat := 98112
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 98110 .coefficient) (.value (.predecessor 1 98111 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98110 .coefficient)
      LeftAuthority98108.bound (LeftAuthority98108.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98108.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98108.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98111 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority98108.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98108.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority98108.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound98112

namespace LeftBound98116
def owner : Owner := ⟨.program ⟨214⟩, ⟨21680⟩⟩
def transferEvent : Nat := 98116
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 98114 .coefficient) (.predecessor 1 98115 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98114 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98115 .coefficient)
      LeftBound98112.bound (LeftBound98112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98113RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98112.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98112.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound98112.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound98112.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound98112.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98116

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
