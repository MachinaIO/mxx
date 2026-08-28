import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard600
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard601

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound87859
def owner : Owner := ⟨.program ⟨214⟩, ⟨24992⟩⟩
def transferEvent : Nat := 87859
def frameStart : Nat := 87747
def rule : BoundRule := .sum [.predecessor 0 87857 .coefficient, .predecessor 1 87858 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87857 .coefficient)
      LeftBound87855.bound (LeftBound87855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87855.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87855.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87858 .coefficient)
      LeftBound87836.bound (LeftBound87836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87841RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87836.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87836.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87855.bound, LeftBound87836.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87855.bound, LeftBound87836.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87855.actual selector witness, LeftBound87836.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87859

namespace LeftBound87872
def owner : Owner := ⟨.program ⟨214⟩, ⟨24990⟩⟩
def transferEvent : Nat := 87872
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 87870 .coefficient, .predecessor 1 87871 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87870 .coefficient)
      LeftBound87695.bound (LeftBound87695.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87695.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87695.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87871 .coefficient)
      LeftBound87678.bound (LeftBound87678.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87685RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87678.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87678.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87695.bound, LeftBound87678.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87695.bound, LeftBound87678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87695.actual selector witness, LeftBound87678.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87872

namespace LeftBound87875
def owner : Owner := ⟨.program ⟨214⟩, ⟨24990⟩⟩
def transferEvent : Nat := 87875
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 87869 .summary, .result 87685 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87869 .summary)
      LeftBound87697.bound (LeftBound87697.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19099⟩⟩) (rawTerms := some (Proof.Events343.exact87869RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87697.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87685 .summary)
      LeftBound87680.bound (LeftBound87680.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24989⟩⟩) (rawTerms := some (Proof.Events342.exact87685RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87680.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87697.bound, LeftBound87680.bound]
def bound : CoeffClass := .finite ⟨352014917316608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87697.bound, LeftBound87680.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87697.actual selector witness, LeftBound87680.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87875

namespace LeftBound87879
def owner : Owner := ⟨.program ⟨214⟩, ⟨26566⟩⟩
def transferEvent : Nat := 87879
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 87877 .coefficient) (.predecessor 1 87878 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87877 .coefficient)
      LeftBound87872.bound (LeftBound87872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87872.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87878 .coefficient)
      LeftAuthority87600.bound (LeftAuthority87600.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87601RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87600.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87600.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound87872.bound LeftAuthority87600.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87872.bound, LeftAuthority87600.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound87872.actual selector witness) * (LeftAuthority87600.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87879

namespace LeftBound87880
def owner : Owner := ⟨.program ⟨214⟩, ⟨26566⟩⟩
def transferEvent : Nat := 87880
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26564⟩⟩]⟩ [⟨.result 87601 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87601 .coefficient)
      LeftAuthority87600.bound (LeftAuthority87600.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26564⟩⟩) (rawTerms := some (Proof.Events342.exact87601RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87600.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87600.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority87600.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87600.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority87600.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound87880

namespace LeftBound87881
def owner : Owner := ⟨.program ⟨214⟩, ⟨26566⟩⟩
def transferEvent : Nat := 87881
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 87876 .summary) (.transfer 87880) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87876 .summary)
      LeftBound87875.bound (LeftBound87875.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24990⟩⟩) (rawTerms := some (Proof.Events343.exact87876RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87875.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 87880)
      LeftBound87880.bound (LeftBound87880.actual selector witness) := by
  exact .transfer (LeftBound87880.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound87875.bound LeftBound87880.bound
def bound : CoeffClass := .finite ⟨1291900378790628425728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87875.bound, LeftBound87880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound87875.actual selector witness) * (LeftBound87880.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87881

namespace LeftBound87892
def owner : Owner := ⟨.program ⟨214⟩, ⟨20538⟩⟩
def transferEvent : Nat := 87892
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 87890 .coefficient) (.value (.predecessor 1 87891 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87890 .coefficient)
      LeftAuthority87888.bound (LeftAuthority87888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87889RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87888.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87888.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87891 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority87888.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87888.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority87888.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound87892

namespace LeftBound87896
def owner : Owner := ⟨.program ⟨214⟩, ⟨20539⟩⟩
def transferEvent : Nat := 87896
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 87894 .coefficient) (.predecessor 1 87895 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87894 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87895 .coefficient)
      LeftBound87892.bound (LeftBound87892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87892.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87892.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound87892.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound87892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound87892.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87896

namespace LeftBound87897
def owner : Owner := ⟨.program ⟨214⟩, ⟨20539⟩⟩
def transferEvent : Nat := 87897
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20536⟩⟩]⟩ [⟨.result 87889 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87889 .coefficient)
      LeftAuthority87888.bound (LeftAuthority87888.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20536⟩⟩) (rawTerms := some (Proof.Events343.exact87889RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87888.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87888.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority87888.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87888.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority87888.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound87897

namespace LeftBound87898
def owner : Owner := ⟨.program ⟨214⟩, ⟨20539⟩⟩
def transferEvent : Nat := 87898
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 87897) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 87897)
      LeftBound87897.bound (LeftBound87897.actual selector witness) := by
  exact .transfer (LeftBound87897.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound87897.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound87897.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound87897.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87898

namespace LeftBound87993
def owner : Owner := ⟨.program ⟨214⟩, ⟨14954⟩⟩
def transferEvent : Nat := 87993
def frameStart : Nat := 87954
def rule : BoundRule := .identity (.predecessor 0 87992 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87992 .coefficient)
      LeftAuthority87990.bound (LeftAuthority87990.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87991RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87990.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87990.derived selector witness)

def rawBound : CoeffClass := LeftAuthority87990.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87990.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority87990.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound87993

namespace LeftBound88010
def owner : Owner := ⟨.program ⟨214⟩, ⟨14993⟩⟩
def transferEvent : Nat := 88010
def frameStart : Nat := 87954
def rule : BoundRule := .sum [.predecessor 0 88008 .coefficient, .predecessor 1 88009 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88008 .coefficient)
      LeftBound87993.bound (LeftBound87993.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound87993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88009 .coefficient)
      LeftAuthority88006.bound (LeftAuthority88006.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority88006.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87993.bound, LeftAuthority88006.bound]
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87993.bound, LeftAuthority88006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87993.actual selector witness, LeftAuthority88006.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88010

namespace LeftBound88013
def owner : Owner := ⟨.program ⟨214⟩, ⟨14994⟩⟩
def transferEvent : Nat := 88013
def frameStart : Nat := 87954
def rule : BoundRule := .identity (.predecessor 0 88012 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88012 .coefficient)
      LeftBound88010.bound (LeftBound88010.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound88010.derived selector witness)

def rawBound : CoeffClass := LeftBound88010.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88010.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound88010.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound88013

namespace LeftBound88019
def owner : Owner := ⟨.program ⟨214⟩, ⟨14995⟩⟩
def transferEvent : Nat := 88019
def frameStart : Nat := 87954
def rule : BoundRule := .product (.predecessor 0 88017 .coefficient) (.predecessor 1 88018 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88017 .coefficient)
      LeftAuthority88015.bound (LeftAuthority88015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact88016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88015.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88018 .coefficient)
      LeftBound88013.bound (LeftBound88013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact88014RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88013.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority88015.bound LeftBound88013.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88015.bound, LeftBound88013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority88015.actual selector witness) * (LeftBound88013.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88019

namespace LeftBound88027
def owner : Owner := ⟨.program ⟨214⟩, ⟨14996⟩⟩
def transferEvent : Nat := 88027
def frameStart : Nat := 87954
def rule : BoundRule := .sum [.predecessor 0 88025 .coefficient, .predecessor 1 88026 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88025 .coefficient)
      LeftAuthority88023.bound (LeftAuthority88023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact88024RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88023.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88023.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88026 .coefficient)
      LeftBound88019.bound (LeftBound88019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact88021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88019.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority88023.bound, LeftBound88019.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88023.bound, LeftBound88019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority88023.actual selector witness, LeftBound88019.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88027

namespace LeftBound88031
def owner : Owner := ⟨.program ⟨214⟩, ⟨26565⟩⟩
def transferEvent : Nat := 88031
def frameStart : Nat := 87954
def rule : BoundRule := .product (.predecessor 0 88029 .coefficient) (.predecessor 1 88030 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88029 .coefficient)
      LeftBound88027.bound (LeftBound88027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact88028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88027.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88030 .coefficient)
      LeftAuthority88004.bound (LeftAuthority88004.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact88005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88004.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88004.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound88027.bound LeftAuthority88004.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88027.bound, LeftAuthority88004.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound88027.actual selector witness) * (LeftAuthority88004.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88031

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
