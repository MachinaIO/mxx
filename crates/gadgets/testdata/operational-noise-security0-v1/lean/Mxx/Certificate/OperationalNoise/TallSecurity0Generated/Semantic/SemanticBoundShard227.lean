import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard196
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard226

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound35041
def owner : Owner := ⟨.program ⟨214⟩, ⟨15167⟩⟩
def transferEvent : Nat := 35041
def frameStart : Nat := 34982
def rule : BoundRule := .identity (.predecessor 0 35040 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35040 .coefficient)
      LeftBound35038.bound (LeftBound35038.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound35038.derived selector witness)

def rawBound : CoeffClass := LeftBound35038.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35038.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound35038.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound35041

namespace LeftBound35047
def owner : Owner := ⟨.program ⟨214⟩, ⟨15168⟩⟩
def transferEvent : Nat := 35047
def frameStart : Nat := 34982
def rule : BoundRule := .product (.predecessor 0 35045 .coefficient) (.predecessor 1 35046 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35045 .coefficient)
      LeftAuthority35043.bound (LeftAuthority35043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events136.exact35044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35043.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35043.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35046 .coefficient)
      LeftBound35041.bound (LeftBound35041.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events136.exact35042RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35041.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35041.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority35043.bound LeftBound35041.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35043.bound, LeftBound35041.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority35043.actual selector witness) * (LeftBound35041.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35047

namespace LeftBound35055
def owner : Owner := ⟨.program ⟨214⟩, ⟨15169⟩⟩
def transferEvent : Nat := 35055
def frameStart : Nat := 34982
def rule : BoundRule := .sum [.predecessor 0 35053 .coefficient, .predecessor 1 35054 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35053 .coefficient)
      LeftAuthority35051.bound (LeftAuthority35051.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events136.exact35052RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35051.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35051.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35054 .coefficient)
      LeftBound35047.bound (LeftBound35047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events136.exact35049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35047.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35047.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority35051.bound, LeftBound35047.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35051.bound, LeftBound35047.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority35051.actual selector witness, LeftBound35047.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35055

namespace LeftBound35059
def owner : Owner := ⟨.program ⟨214⟩, ⟨26814⟩⟩
def transferEvent : Nat := 35059
def frameStart : Nat := 34982
def rule : BoundRule := .product (.predecessor 0 35057 .coefficient) (.predecessor 1 35058 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35057 .coefficient)
      LeftBound35055.bound (LeftBound35055.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events136.exact35056RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35055.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35055.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35058 .coefficient)
      LeftAuthority35032.bound (LeftAuthority35032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events136.exact35033RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35032.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35032.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound35055.bound LeftAuthority35032.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35055.bound, LeftAuthority35032.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound35055.actual selector witness) * (LeftAuthority35032.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35059

namespace LeftBound35070
def owner : Owner := ⟨.program ⟨214⟩, ⟨15226⟩⟩
def transferEvent : Nat := 35070
def frameStart : Nat := 34982
def rule : BoundRule := .product (.predecessor 0 35068 .coefficient) (.predecessor 1 35069 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35068 .coefficient)
      LeftAuthority35043.bound (LeftAuthority35043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events136.exact35044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35043.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35043.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35069 .coefficient)
      LeftAuthority35066.bound (LeftAuthority35066.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events136.exact35067RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35066.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35066.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority35043.bound LeftAuthority35066.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35043.bound, LeftAuthority35066.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority35043.actual selector witness) * (LeftAuthority35066.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35070

namespace LeftBound35078
def owner : Owner := ⟨.program ⟨214⟩, ⟨15227⟩⟩
def transferEvent : Nat := 35078
def frameStart : Nat := 34982
def rule : BoundRule := .sum [.predecessor 0 35076 .coefficient, .predecessor 1 35077 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35076 .coefficient)
      LeftAuthority35074.bound (LeftAuthority35074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35074.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35074.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35077 .coefficient)
      LeftBound35070.bound (LeftBound35070.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35070.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35070.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority35074.bound, LeftBound35070.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35074.bound, LeftBound35070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority35074.actual selector witness, LeftBound35070.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35078

namespace LeftBound35082
def owner : Owner := ⟨.program ⟨214⟩, ⟨26819⟩⟩
def transferEvent : Nat := 35082
def frameStart : Nat := 34982
def rule : BoundRule := .sum [.predecessor 0 35080 .coefficient, .predecessor 1 35081 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35080 .coefficient)
      LeftBound35078.bound (LeftBound35078.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35079RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35078.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35078.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35081 .coefficient)
      LeftBound35059.bound (LeftBound35059.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events136.exact35064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35059.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35059.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35078.bound, LeftBound35059.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35078.bound, LeftBound35059.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35078.actual selector witness, LeftBound35059.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35082

namespace LeftBound35095
def owner : Owner := ⟨.program ⟨214⟩, ⟨26816⟩⟩
def transferEvent : Nat := 35095
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35093 .coefficient, .predecessor 1 35094 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35093 .coefficient)
      LeftBound34924.bound (LeftBound34924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35092RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34924.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34924.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35094 .coefficient)
      LeftBound34907.bound (LeftBound34907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events136.exact34914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34907.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound34924.bound, LeftBound34907.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34924.bound, LeftBound34907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound34924.actual selector witness, LeftBound34907.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35095

namespace LeftBound35098
def owner : Owner := ⟨.program ⟨214⟩, ⟨26816⟩⟩
def transferEvent : Nat := 35098
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35092 .summary, .result 34914 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35092 .summary)
      LeftBound34926.bound (LeftBound34926.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20623⟩⟩) (rawTerms := some (Proof.Events137.exact35092RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34926.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 34914 .summary)
      LeftBound34909.bound (LeftBound34909.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26815⟩⟩) (rawTerms := some (Proof.Events136.exact34914RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34909.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound34926.bound, LeftBound34909.bound]
def bound : CoeffClass := .finite ⟨1291911586824442228736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34926.bound, LeftBound34909.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound34926.actual selector witness, LeftBound34909.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35098

namespace LeftBound35102
def owner : Owner := ⟨.program ⟨214⟩, ⟨26817⟩⟩
def transferEvent : Nat := 35102
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 35100 .coefficient) (.predecessor 1 35101 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35100 .coefficient)
      LeftBound35095.bound (LeftBound35095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35099RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35095.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35095.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35101 .coefficient)
      LeftBound5818.bound (LeftBound5818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5819RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5818.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5818.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound35095.bound LeftBound5818.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35095.bound, LeftBound5818.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound35095.actual selector witness) * (LeftBound5818.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35102

namespace LeftBound35103
def owner : Owner := ⟨.program ⟨214⟩, ⟨26817⟩⟩
def transferEvent : Nat := 35103
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩ [⟨.result 5815 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5815 .coefficient)
      LeftAuthority5814.bound (LeftAuthority5814.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6663⟩⟩) (rawTerms := some (Proof.Events022.exact5815RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5814.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5814.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5814.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5814.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5814.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound35103

namespace LeftBound35104
def owner : Owner := ⟨.program ⟨214⟩, ⟨26817⟩⟩
def transferEvent : Nat := 35104
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 35099 .summary) (.transfer 35103) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35099 .summary)
      LeftBound35098.bound (LeftBound35098.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26816⟩⟩) (rawTerms := some (Proof.Events137.exact35099RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35098.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35103)
      LeftBound35103.bound (LeftBound35103.actual selector witness) := by
  exact .transfer (LeftBound35103.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound35098.bound LeftBound35103.bound
def bound : CoeffClass := .finite ⟨4741336194231092170536779776, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35098.bound, LeftBound35103.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound35098.actual selector witness) * (LeftBound35103.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35104

namespace LeftBound35119
def owner : Owner := ⟨.program ⟨214⟩, ⟨26598⟩⟩
def transferEvent : Nat := 35119
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 35117 .coefficient) (.predecessor 1 35118 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35117 .coefficient)
      LeftBound29406.bound (LeftBound29406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29406.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29406.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35118 .coefficient)
      LeftAuthority35115.bound (LeftAuthority35115.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35116RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35115.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35115.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29406.bound LeftAuthority35115.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29406.bound, LeftAuthority35115.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29406.actual selector witness) * (LeftAuthority35115.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35119

namespace LeftBound35120
def owner : Owner := ⟨.program ⟨214⟩, ⟨26598⟩⟩
def transferEvent : Nat := 35120
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26596⟩⟩]⟩ [⟨.result 35116 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35116 .coefficient)
      LeftAuthority35115.bound (LeftAuthority35115.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26596⟩⟩) (rawTerms := some (Proof.Events137.exact35116RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35115.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35115.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority35115.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35115.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority35115.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound35120

namespace LeftBound35121
def owner : Owner := ⟨.program ⟨214⟩, ⟨26598⟩⟩
def transferEvent : Nat := 35121
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 29410 .summary) (.transfer 35120) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29410 .summary)
      LeftBound29409.bound (LeftBound29409.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25005⟩⟩) (rawTerms := some (Proof.Events114.exact29410RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29409.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35120)
      LeftBound35120.bound (LeftBound35120.actual selector witness) := by
  exact .transfer (LeftBound35120.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29409.bound LeftBound35120.bound
def bound : CoeffClass := .finite ⟨1291900378790628425728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29409.bound, LeftBound35120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29409.actual selector witness) * (LeftBound35120.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35121

namespace LeftBound35132
def owner : Owner := ⟨.program ⟨214⟩, ⟨20478⟩⟩
def transferEvent : Nat := 35132
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 35130 .coefficient) (.value (.predecessor 1 35131 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35130 .coefficient)
      LeftAuthority35128.bound (LeftAuthority35128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35128.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35128.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35131 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority35128.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35128.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority35128.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound35132

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
