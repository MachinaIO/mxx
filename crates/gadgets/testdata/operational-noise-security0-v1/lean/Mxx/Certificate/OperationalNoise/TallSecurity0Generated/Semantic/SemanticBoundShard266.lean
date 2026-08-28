import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard060
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard061
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard237
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard265

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound39852
def owner : Owner := ⟨.program ⟨214⟩, ⟨28544⟩⟩
def transferEvent : Nat := 39852
def frameStart : Nat := 39775
def rule : BoundRule := .product (.predecessor 0 39850 .coefficient) (.predecessor 1 39851 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39850 .coefficient)
      LeftBound39848.bound (LeftBound39848.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39849RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39848.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39848.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39851 .coefficient)
      LeftAuthority39825.bound (LeftAuthority39825.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39826RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39825.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39825.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound39848.bound LeftAuthority39825.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39848.bound, LeftAuthority39825.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound39848.actual selector witness) * (LeftAuthority39825.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39852

namespace LeftBound39863
def owner : Owner := ⟨.program ⟨214⟩, ⟨16315⟩⟩
def transferEvent : Nat := 39863
def frameStart : Nat := 39775
def rule : BoundRule := .product (.predecessor 0 39861 .coefficient) (.predecessor 1 39862 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39861 .coefficient)
      LeftAuthority39836.bound (LeftAuthority39836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39837RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39836.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39836.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39862 .coefficient)
      LeftAuthority39859.bound (LeftAuthority39859.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39859.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39859.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority39836.bound LeftAuthority39859.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39836.bound, LeftAuthority39859.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority39836.actual selector witness) * (LeftAuthority39859.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39863

namespace LeftBound39871
def owner : Owner := ⟨.program ⟨214⟩, ⟨16316⟩⟩
def transferEvent : Nat := 39871
def frameStart : Nat := 39775
def rule : BoundRule := .sum [.predecessor 0 39869 .coefficient, .predecessor 1 39870 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39869 .coefficient)
      LeftAuthority39867.bound (LeftAuthority39867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39867.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39867.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39870 .coefficient)
      LeftBound39863.bound (LeftBound39863.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39863.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39863.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority39867.bound, LeftBound39863.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39867.bound, LeftBound39863.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority39867.actual selector witness, LeftBound39863.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39871

namespace LeftBound39875
def owner : Owner := ⟨.program ⟨214⟩, ⟨28548⟩⟩
def transferEvent : Nat := 39875
def frameStart : Nat := 39775
def rule : BoundRule := .sum [.predecessor 0 39873 .coefficient, .predecessor 1 39874 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39873 .coefficient)
      LeftBound39871.bound (LeftBound39871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39871.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39874 .coefficient)
      LeftBound39852.bound (LeftBound39852.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39857RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39852.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39852.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39871.bound, LeftBound39852.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39871.bound, LeftBound39852.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39871.actual selector witness, LeftBound39852.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39875

namespace LeftBound39888
def owner : Owner := ⟨.program ⟨214⟩, ⟨28546⟩⟩
def transferEvent : Nat := 39888
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 39886 .coefficient, .predecessor 1 39887 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39886 .coefficient)
      LeftBound39717.bound (LeftBound39717.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39885RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39717.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39717.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39887 .coefficient)
      LeftBound39700.bound (LeftBound39700.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39707RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39700.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39700.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39717.bound, LeftBound39700.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39717.bound, LeftBound39700.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39717.actual selector witness, LeftBound39700.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39888

namespace LeftBound39891
def owner : Owner := ⟨.program ⟨214⟩, ⟨28546⟩⟩
def transferEvent : Nat := 39891
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 39885 .summary, .result 39707 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39885 .summary)
      LeftBound39719.bound (LeftBound39719.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21843⟩⟩) (rawTerms := some (Proof.Events155.exact39885RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39719.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39707 .summary)
      LeftBound39702.bound (LeftBound39702.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28545⟩⟩) (rawTerms := some (Proof.Events155.exact39707RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39702.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39719.bound, LeftBound39702.bound]
def bound : CoeffClass := .finite ⟨1292202948609709846528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39719.bound, LeftBound39702.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39719.actual selector witness, LeftBound39702.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39891

namespace LeftBound39915
def owner : Owner := ⟨.program ⟨214⟩, ⟨11646⟩⟩
def transferEvent : Nat := 39915
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 39913 .coefficient) (.predecessor 1 39914 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39913 .coefficient)
      LeftAuthority1773.bound (LeftAuthority1773.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1774RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1773.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1773.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39914 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1773.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1773.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1773.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound39915

namespace LeftBound39920
def owner : Owner := ⟨.program ⟨214⟩, ⟨7313⟩⟩
def transferEvent : Nat := 39920
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 39918 .coefficient) (.predecessor 1 39919 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39918 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39919 .coefficient)
      LeftBound10479.bound (LeftBound10479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10479.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound10479.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound10479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound10479.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39920

namespace LeftBound39925
def owner : Owner := ⟨.program ⟨214⟩, ⟨11647⟩⟩
def transferEvent : Nat := 39925
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 39923 .coefficient, .predecessor 1 39924 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39923 .coefficient)
      LeftBound39920.bound (LeftBound39920.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39922RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39920.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39920.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39924 .coefficient)
      LeftBound39915.bound (LeftBound39915.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39917RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39915.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39915.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39920.bound, LeftBound39915.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39920.bound, LeftBound39915.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39920.actual selector witness, LeftBound39915.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39925

namespace LeftBound39929
def owner : Owner := ⟨.program ⟨214⟩, ⟨11648⟩⟩
def transferEvent : Nat := 39929
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 39927 .coefficient, .predecessor 1 39928 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39927 .coefficient)
      LeftBound39925.bound (LeftBound39925.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39926RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39925.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39925.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39928 .coefficient)
      LeftBound10471.bound (LeftBound10471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10471.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39925.bound, LeftBound10471.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39925.bound, LeftBound10471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39925.actual selector witness, LeftBound10471.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39929

namespace LeftBound39930
def owner : Owner := ⟨.program ⟨214⟩, ⟨11648⟩⟩
def transferEvent : Nat := 39930
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨95⟩⟩]⟩ [⟨.result 10472 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10472 .coefficient)
      LeftBound10471.bound (LeftBound10471.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨95⟩⟩) (rawTerms := some (Proof.Events040.exact10472RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10471.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound10471.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound10471.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound39930

namespace LeftBound39935
def owner : Owner := ⟨.program ⟨214⟩, ⟨14662⟩⟩
def transferEvent : Nat := 39935
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 39933 .coefficient) (.predecessor 1 39934 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39933 .coefficient)
      LeftBound39929.bound (LeftBound39929.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39929.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39929.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39934 .coefficient)
      LeftAuthority1776.bound (LeftAuthority1776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1777RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1776.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1776.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound39929.bound LeftAuthority1776.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39929.bound, LeftAuthority1776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound39929.actual selector witness) * (LeftAuthority1776.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39935

namespace LeftBound39936
def owner : Owner := ⟨.program ⟨214⟩, ⟨14662⟩⟩
def transferEvent : Nat := 39936
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨14659⟩⟩], []⟩ [⟨.result 1777 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 1777 .coefficient)
      LeftAuthority1776.bound (LeftAuthority1776.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨14659⟩⟩) (rawTerms := some (Proof.Events006.exact1777RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1776.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1776.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1776.bound []
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority1776.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound39936

namespace LeftBound39937
def owner : Owner := ⟨.program ⟨214⟩, ⟨14662⟩⟩
def transferEvent : Nat := 39937
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 39932 .summary) (.transfer 39936) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39932 .summary)
      LeftBound39930.bound (LeftBound39930.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11648⟩⟩) (rawTerms := some (Proof.Events155.exact39932RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39930.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 39936)
      LeftBound39936.bound (LeftBound39936.actual selector witness) := by
  exact .transfer (LeftBound39936.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound39930.bound LeftBound39936.bound
def bound : CoeffClass := .finite ⟨23296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39930.bound, LeftBound39936.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound39930.actual selector witness) * (LeftBound39936.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39937

namespace LeftBound39943
def owner : Owner := ⟨.program ⟨214⟩, ⟨14663⟩⟩
def transferEvent : Nat := 39943
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 39941 .coefficient) (.predecessor 1 39942 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39941 .coefficient)
      LeftAuthority1776.bound (LeftAuthority1776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1777RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1776.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1776.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39942 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1776.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1776.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1776.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound39943

namespace LeftBound39948
def owner : Owner := ⟨.program ⟨214⟩, ⟨7294⟩⟩
def transferEvent : Nat := 39948
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 39946 .coefficient) (.predecessor 1 39947 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39946 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39947 .coefficient)
      LeftBound10520.bound (LeftBound10520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10520.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound10520.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound10520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound10520.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39948

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
