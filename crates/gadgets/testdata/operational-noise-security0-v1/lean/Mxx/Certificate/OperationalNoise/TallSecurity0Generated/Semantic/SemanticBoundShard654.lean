import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard044
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard653

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound96019
def owner : Owner := ⟨.program ⟨214⟩, ⟨16624⟩⟩
def transferEvent : Nat := 96019
def frameStart : Nat := 95992
def rule : BoundRule := .identity (.predecessor 0 96018 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96018 .coefficient)
      LeftAuthority96016.bound (LeftAuthority96016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96016.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96016.derived selector witness)

def rawBound : CoeffClass := LeftAuthority96016.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority96016.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound96019

namespace LeftBound96036
def owner : Owner := ⟨.program ⟨214⟩, ⟨16700⟩⟩
def transferEvent : Nat := 96036
def frameStart : Nat := 95992
def rule : BoundRule := .sum [.predecessor 0 96034 .coefficient, .predecessor 1 96035 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96034 .coefficient)
      LeftBound96019.bound (LeftBound96019.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound96019.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96035 .coefficient)
      LeftAuthority96032.bound (LeftAuthority96032.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority96032.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96019.bound, LeftAuthority96032.bound]
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96019.bound, LeftAuthority96032.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96019.actual selector witness, LeftAuthority96032.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96036

namespace LeftBound96039
def owner : Owner := ⟨.program ⟨214⟩, ⟨16701⟩⟩
def transferEvent : Nat := 96039
def frameStart : Nat := 95992
def rule : BoundRule := .identity (.predecessor 0 96038 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96038 .coefficient)
      LeftBound96036.bound (LeftBound96036.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound96036.derived selector witness)

def rawBound : CoeffClass := LeftBound96036.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96036.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound96036.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound96039

namespace LeftBound96045
def owner : Owner := ⟨.program ⟨214⟩, ⟨16702⟩⟩
def transferEvent : Nat := 96045
def frameStart : Nat := 95992
def rule : BoundRule := .product (.predecessor 0 96043 .coefficient) (.predecessor 1 96044 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96043 .coefficient)
      LeftAuthority96041.bound (LeftAuthority96041.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96042RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96041.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96041.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96044 .coefficient)
      LeftBound96039.bound (LeftBound96039.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96040RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96039.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96039.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority96041.bound LeftBound96039.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96041.bound, LeftBound96039.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority96041.actual selector witness) * (LeftBound96039.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96045

namespace LeftBound96053
def owner : Owner := ⟨.program ⟨214⟩, ⟨16703⟩⟩
def transferEvent : Nat := 96053
def frameStart : Nat := 95992
def rule : BoundRule := .sum [.predecessor 0 96051 .coefficient, .predecessor 1 96052 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96051 .coefficient)
      LeftAuthority96049.bound (LeftAuthority96049.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96050RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96049.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96049.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96052 .coefficient)
      LeftBound96045.bound (LeftBound96045.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96047RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96045.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96045.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority96049.bound, LeftBound96045.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96049.bound, LeftBound96045.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority96049.actual selector witness, LeftBound96045.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96053

namespace LeftBound96057
def owner : Owner := ⟨.program ⟨214⟩, ⟨29351⟩⟩
def transferEvent : Nat := 96057
def frameStart : Nat := 95992
def rule : BoundRule := .product (.predecessor 0 96055 .coefficient) (.predecessor 1 96056 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96055 .coefficient)
      LeftBound96053.bound (LeftBound96053.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96054RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96053.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96053.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96056 .coefficient)
      LeftAuthority96030.bound (LeftAuthority96030.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96031RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96030.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96030.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound96053.bound LeftAuthority96030.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96053.bound, LeftAuthority96030.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound96053.actual selector witness) * (LeftAuthority96030.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96057

namespace LeftBound96068
def owner : Owner := ⟨.program ⟨214⟩, ⟨16673⟩⟩
def transferEvent : Nat := 96068
def frameStart : Nat := 95992
def rule : BoundRule := .product (.predecessor 0 96066 .coefficient) (.predecessor 1 96067 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96066 .coefficient)
      LeftAuthority96041.bound (LeftAuthority96041.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96042RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96041.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96041.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96067 .coefficient)
      LeftAuthority96064.bound (LeftAuthority96064.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96065RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96064.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96064.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority96041.bound LeftAuthority96064.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96041.bound, LeftAuthority96064.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority96041.actual selector witness) * (LeftAuthority96064.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96068

namespace LeftBound96076
def owner : Owner := ⟨.program ⟨214⟩, ⟨16674⟩⟩
def transferEvent : Nat := 96076
def frameStart : Nat := 95992
def rule : BoundRule := .sum [.predecessor 0 96074 .coefficient, .predecessor 1 96075 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96074 .coefficient)
      LeftAuthority96072.bound (LeftAuthority96072.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96073RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96072.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96072.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96075 .coefficient)
      LeftBound96068.bound (LeftBound96068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96070RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96068.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96068.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority96072.bound, LeftBound96068.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96072.bound, LeftBound96068.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority96072.actual selector witness, LeftBound96068.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96076

namespace LeftBound96080
def owner : Owner := ⟨.program ⟨214⟩, ⟨29355⟩⟩
def transferEvent : Nat := 96080
def frameStart : Nat := 95992
def rule : BoundRule := .sum [.predecessor 0 96078 .coefficient, .predecessor 1 96079 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96078 .coefficient)
      LeftBound96076.bound (LeftBound96076.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96077RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96076.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96076.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96079 .coefficient)
      LeftBound96057.bound (LeftBound96057.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96062RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96057.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96057.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96076.bound, LeftBound96057.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96076.bound, LeftBound96057.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96076.actual selector witness, LeftBound96057.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96080

namespace LeftBound96093
def owner : Owner := ⟨.program ⟨214⟩, ⟨29353⟩⟩
def transferEvent : Nat := 96093
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 96091 .coefficient, .predecessor 1 96092 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96091 .coefficient)
      LeftBound95946.bound (LeftBound95946.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96090RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95946.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95946.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96092 .coefficient)
      LeftBound95929.bound (LeftBound95929.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events374.exact95936RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95929.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95929.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95946.bound, LeftBound95929.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95946.bound, LeftBound95929.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95946.actual selector witness, LeftBound95929.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96093

namespace LeftBound96096
def owner : Owner := ⟨.program ⟨214⟩, ⟨29353⟩⟩
def transferEvent : Nat := 96096
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 96090 .summary, .result 95936 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96090 .summary)
      LeftBound95948.bound (LeftBound95948.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22400⟩⟩) (rawTerms := some (Proof.Events375.exact96090RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95948.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95936 .summary)
      LeftBound95931.bound (LeftBound95931.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29352⟩⟩) (rawTerms := some (Proof.Events374.exact95936RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95931.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95948.bound, LeftBound95931.bound]
def bound : CoeffClass := .finite ⟨1292382248169874534400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95948.bound, LeftBound95931.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95948.actual selector witness, LeftBound95931.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96096

namespace LeftBound96120
def owner : Owner := ⟨.program ⟨214⟩, ⟨12545⟩⟩
def transferEvent : Nat := 96120
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 96118 .coefficient) (.predecessor 1 96119 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96118 .coefficient)
      LeftAuthority4657.bound (LeftAuthority4657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4658RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4657.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4657.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96119 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4657.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4657.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4657.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound96120

namespace LeftBound96125
def owner : Owner := ⟨.program ⟨214⟩, ⟨7123⟩⟩
def transferEvent : Nat := 96125
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 96123 .coefficient) (.predecessor 1 96124 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96123 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96124 .coefficient)
      LeftBound8475.bound (LeftBound8475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8475.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound8475.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound8475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound8475.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96125

namespace LeftBound96130
def owner : Owner := ⟨.program ⟨214⟩, ⟨12546⟩⟩
def transferEvent : Nat := 96130
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 96128 .coefficient, .predecessor 1 96129 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96128 .coefficient)
      LeftBound96125.bound (LeftBound96125.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96127RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96125.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96125.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96129 .coefficient)
      LeftBound96120.bound (LeftBound96120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96122RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96120.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96120.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96125.bound, LeftBound96120.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96125.bound, LeftBound96120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96125.actual selector witness, LeftBound96120.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96130

namespace LeftBound96134
def owner : Owner := ⟨.program ⟨214⟩, ⟨12547⟩⟩
def transferEvent : Nat := 96134
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 96132 .coefficient, .predecessor 1 96133 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96132 .coefficient)
      LeftBound96130.bound (LeftBound96130.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96131RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96130.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96130.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96133 .coefficient)
      LeftBound8467.bound (LeftBound8467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8467.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96130.bound, LeftBound8467.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96130.bound, LeftBound8467.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96130.actual selector witness, LeftBound8467.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96134

namespace LeftBound96135
def owner : Owner := ⟨.program ⟨214⟩, ⟨12547⟩⟩
def transferEvent : Nat := 96135
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨100⟩⟩]⟩ [⟨.result 8468 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8468 .coefficient)
      LeftBound8467.bound (LeftBound8467.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨100⟩⟩) (rawTerms := some (Proof.Events033.exact8468RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8467.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound8467.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8467.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound8467.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound96135

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
