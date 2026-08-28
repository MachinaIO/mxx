import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard474

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound69871
def owner : Owner := ⟨.program ⟨214⟩, ⟨26141⟩⟩
def transferEvent : Nat := 69871
def frameStart : Nat := 69780
def rule : BoundRule := .product (.predecessor 0 69869 .coefficient) (.predecessor 1 69870 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69869 .coefficient)
      LeftBound69867.bound (LeftBound69867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69867.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69867.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69870 .coefficient)
      LeftAuthority69824.bound (LeftAuthority69824.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69825RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69824.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69824.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound69867.bound LeftAuthority69824.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69867.bound, LeftAuthority69824.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound69867.actual selector witness) * (LeftAuthority69824.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69871

namespace LeftBound69882
def owner : Owner := ⟨.program ⟨214⟩, ⟨16057⟩⟩
def transferEvent : Nat := 69882
def frameStart : Nat := 69780
def rule : BoundRule := .product (.predecessor 0 69880 .coefficient) (.predecessor 1 69881 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69880 .coefficient)
      LeftAuthority69835.bound (LeftAuthority69835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69835.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69835.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69881 .coefficient)
      LeftAuthority69878.bound (LeftAuthority69878.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69879RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69878.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69878.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority69835.bound LeftAuthority69878.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69835.bound, LeftAuthority69878.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority69835.actual selector witness) * (LeftAuthority69878.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69882

namespace LeftBound69890
def owner : Owner := ⟨.program ⟨214⟩, ⟨16058⟩⟩
def transferEvent : Nat := 69890
def frameStart : Nat := 69780
def rule : BoundRule := .sum [.predecessor 0 69888 .coefficient, .predecessor 1 69889 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69888 .coefficient)
      LeftAuthority69886.bound (LeftAuthority69886.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69887RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69886.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69886.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69889 .coefficient)
      LeftBound69882.bound (LeftBound69882.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69882.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69882.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority69886.bound, LeftBound69882.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69886.bound, LeftBound69882.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority69886.actual selector witness, LeftBound69882.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69890

namespace LeftBound69894
def owner : Owner := ⟨.program ⟨214⟩, ⟨26142⟩⟩
def transferEvent : Nat := 69894
def frameStart : Nat := 69780
def rule : BoundRule := .sum [.predecessor 0 69892 .coefficient, .predecessor 1 69893 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69892 .coefficient)
      LeftBound69890.bound (LeftBound69890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69890.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69890.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69893 .coefficient)
      LeftBound69871.bound (LeftBound69871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69871.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69871.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69890.bound, LeftBound69871.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69890.bound, LeftBound69871.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound69890.actual selector witness, LeftBound69871.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69894

namespace LeftBound69907
def owner : Owner := ⟨.program ⟨214⟩, ⟨26140⟩⟩
def transferEvent : Nat := 69907
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 69905 .coefficient, .predecessor 1 69906 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69905 .coefficient)
      LeftBound69728.bound (LeftBound69728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69904RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69728.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69906 .coefficient)
      LeftBound69711.bound (LeftBound69711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69718RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69711.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69711.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69728.bound, LeftBound69711.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69728.bound, LeftBound69711.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound69728.actual selector witness, LeftBound69711.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69907

namespace LeftBound69910
def owner : Owner := ⟨.program ⟨214⟩, ⟨26140⟩⟩
def transferEvent : Nat := 69910
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 69904 .summary, .result 69718 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 69904 .summary)
      LeftBound69730.bound (LeftBound69730.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19599⟩⟩) (rawTerms := some (Proof.Events273.exact69904RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69730.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 69718 .summary)
      LeftBound69713.bound (LeftBound69713.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26139⟩⟩) (rawTerms := some (Proof.Events272.exact69718RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69713.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69730.bound, LeftBound69713.bound]
def bound : CoeffClass := .finite ⟨352072932929536, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69730.bound, LeftBound69713.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound69730.actual selector witness, LeftBound69713.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69910

namespace LeftBound69914
def owner : Owner := ⟨.program ⟨214⟩, ⟨28072⟩⟩
def transferEvent : Nat := 69914
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 69912 .coefficient) (.predecessor 1 69913 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69912 .coefficient)
      LeftBound69907.bound (LeftBound69907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69907.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69913 .coefficient)
      LeftAuthority69633.bound (LeftAuthority69633.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69634RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69633.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69633.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound69907.bound LeftAuthority69633.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69907.bound, LeftAuthority69633.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound69907.actual selector witness) * (LeftAuthority69633.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69914

namespace LeftBound69915
def owner : Owner := ⟨.program ⟨214⟩, ⟨28072⟩⟩
def transferEvent : Nat := 69915
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28070⟩⟩]⟩ [⟨.result 69634 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 69634 .coefficient)
      LeftAuthority69633.bound (LeftAuthority69633.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28070⟩⟩) (rawTerms := some (Proof.Events272.exact69634RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69633.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69633.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority69633.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69633.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority69633.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound69915

namespace LeftBound69916
def owner : Owner := ⟨.program ⟨214⟩, ⟨28072⟩⟩
def transferEvent : Nat := 69916
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 69911 .summary) (.transfer 69915) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 69911 .summary)
      LeftBound69910.bound (LeftBound69910.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26140⟩⟩) (rawTerms := some (Proof.Events273.exact69911RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69910.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 69915)
      LeftBound69915.bound (LeftBound69915.actual selector witness) := by
  exact .transfer (LeftBound69915.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound69910.bound LeftBound69915.bound
def bound : CoeffClass := .finite ⟨1292113297018323992576, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69910.bound, LeftBound69915.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound69910.actual selector witness) * (LeftBound69915.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69916

namespace LeftBound69927
def owner : Owner := ⟨.program ⟨214⟩, ⟨21542⟩⟩
def transferEvent : Nat := 69927
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 69925 .coefficient) (.value (.predecessor 1 69926 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69925 .coefficient)
      LeftAuthority69923.bound (LeftAuthority69923.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69923.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69923.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69926 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority69923.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69923.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority69923.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound69927

namespace LeftBound69931
def owner : Owner := ⟨.program ⟨214⟩, ⟨21543⟩⟩
def transferEvent : Nat := 69931
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 69929 .coefficient) (.predecessor 1 69930 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69929 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69930 .coefficient)
      LeftBound69927.bound (LeftBound69927.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69927.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69927.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound69927.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound69927.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound69927.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69931

namespace LeftBound69932
def owner : Owner := ⟨.program ⟨214⟩, ⟨21543⟩⟩
def transferEvent : Nat := 69932
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21540⟩⟩]⟩ [⟨.result 69924 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 69924 .coefficient)
      LeftAuthority69923.bound (LeftAuthority69923.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21540⟩⟩) (rawTerms := some (Proof.Events273.exact69924RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69923.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69923.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority69923.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69923.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority69923.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound69932

namespace LeftBound69933
def owner : Owner := ⟨.program ⟨214⟩, ⟨21543⟩⟩
def transferEvent : Nat := 69933
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 69932) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 69932)
      LeftBound69932.bound (LeftBound69932.actual selector witness) := by
  exact .transfer (LeftBound69932.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound69932.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound69932.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound69932.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69933

namespace LeftBound70028
def owner : Owner := ⟨.program ⟨214⟩, ⟨16056⟩⟩
def transferEvent : Nat := 70028
def frameStart : Nat := 69989
def rule : BoundRule := .identity (.predecessor 0 70027 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70027 .coefficient)
      LeftAuthority70025.bound (LeftAuthority70025.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70026RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70025.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70025.derived selector witness)

def rawBound : CoeffClass := LeftAuthority70025.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70025.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority70025.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound70028

namespace LeftBound70045
def owner : Owner := ⟨.program ⟨214⟩, ⟨16130⟩⟩
def transferEvent : Nat := 70045
def frameStart : Nat := 69989
def rule : BoundRule := .sum [.predecessor 0 70043 .coefficient, .predecessor 1 70044 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70043 .coefficient)
      LeftBound70028.bound (LeftBound70028.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound70028.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70044 .coefficient)
      LeftAuthority70041.bound (LeftAuthority70041.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority70041.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70028.bound, LeftAuthority70041.bound]
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70028.bound, LeftAuthority70041.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70028.actual selector witness, LeftAuthority70041.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70045

namespace LeftBound70048
def owner : Owner := ⟨.program ⟨214⟩, ⟨16131⟩⟩
def transferEvent : Nat := 70048
def frameStart : Nat := 69989
def rule : BoundRule := .identity (.predecessor 0 70047 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70047 .coefficient)
      LeftBound70045.bound (LeftBound70045.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound70045.derived selector witness)

def rawBound : CoeffClass := LeftBound70045.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70045.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound70045.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound70048

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
