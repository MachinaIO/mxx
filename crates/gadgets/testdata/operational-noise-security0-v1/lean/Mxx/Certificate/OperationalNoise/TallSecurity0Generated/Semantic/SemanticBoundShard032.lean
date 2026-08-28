import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard028
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard031

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound6769
def owner : Owner := ⟨.program ⟨214⟩, ⟨22859⟩⟩
def transferEvent : Nat := 6769
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 6768) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6768)
      LeftBound6768.bound (LeftBound6768.actual selector witness) := by
  exact .transfer (LeftBound6768.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound6768.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound6768.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound6768.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6769

namespace LeftBound6864
def owner : Owner := ⟨.program ⟨214⟩, ⟨17028⟩⟩
def transferEvent : Nat := 6864
def frameStart : Nat := 6825
def rule : BoundRule := .identity (.predecessor 0 6863 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6863 .coefficient)
      LeftAuthority6861.bound (LeftAuthority6861.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6862RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6861.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6861.derived selector witness)

def rawBound : CoeffClass := LeftAuthority6861.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6861.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority6861.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound6864

namespace LeftBound6881
def owner : Owner := ⟨.program ⟨214⟩, ⟨17067⟩⟩
def transferEvent : Nat := 6881
def frameStart : Nat := 6825
def rule : BoundRule := .sum [.predecessor 0 6879 .coefficient, .predecessor 1 6880 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6879 .coefficient)
      LeftBound6864.bound (LeftBound6864.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound6864.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6880 .coefficient)
      LeftAuthority6877.bound (LeftAuthority6877.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority6877.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6864.bound, LeftAuthority6877.bound]
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6864.bound, LeftAuthority6877.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6864.actual selector witness, LeftAuthority6877.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6881

namespace LeftBound6884
def owner : Owner := ⟨.program ⟨214⟩, ⟨17068⟩⟩
def transferEvent : Nat := 6884
def frameStart : Nat := 6825
def rule : BoundRule := .identity (.predecessor 0 6883 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6883 .coefficient)
      LeftBound6881.bound (LeftBound6881.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound6881.derived selector witness)

def rawBound : CoeffClass := LeftBound6881.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound6881.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound6884

namespace LeftBound6890
def owner : Owner := ⟨.program ⟨214⟩, ⟨17069⟩⟩
def transferEvent : Nat := 6890
def frameStart : Nat := 6825
def rule : BoundRule := .product (.predecessor 0 6888 .coefficient) (.predecessor 1 6889 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6888 .coefficient)
      LeftAuthority6886.bound (LeftAuthority6886.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6887RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6886.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6886.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6889 .coefficient)
      LeftBound6884.bound (LeftBound6884.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6885RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6884.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6884.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority6886.bound LeftBound6884.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6886.bound, LeftBound6884.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority6886.actual selector witness) * (LeftBound6884.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6890

namespace LeftBound6898
def owner : Owner := ⟨.program ⟨214⟩, ⟨17070⟩⟩
def transferEvent : Nat := 6898
def frameStart : Nat := 6825
def rule : BoundRule := .sum [.predecessor 0 6896 .coefficient, .predecessor 1 6897 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6896 .coefficient)
      LeftAuthority6894.bound (LeftAuthority6894.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6894.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6894.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6897 .coefficient)
      LeftBound6890.bound (LeftBound6890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6890.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6890.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority6894.bound, LeftBound6890.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6894.bound, LeftBound6890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority6894.actual selector witness, LeftBound6890.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6898

namespace LeftBound6902
def owner : Owner := ⟨.program ⟨214⟩, ⟨30206⟩⟩
def transferEvent : Nat := 6902
def frameStart : Nat := 6825
def rule : BoundRule := .product (.predecessor 0 6900 .coefficient) (.predecessor 1 6901 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6900 .coefficient)
      LeftBound6898.bound (LeftBound6898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6898.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6901 .coefficient)
      LeftAuthority6875.bound (LeftAuthority6875.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6875.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6875.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound6898.bound LeftAuthority6875.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6898.bound, LeftAuthority6875.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound6898.actual selector witness) * (LeftAuthority6875.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6902

namespace LeftBound6913
def owner : Owner := ⟨.program ⟨214⟩, ⟨18183⟩⟩
def transferEvent : Nat := 6913
def frameStart : Nat := 6825
def rule : BoundRule := .product (.predecessor 0 6911 .coefficient) (.predecessor 1 6912 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6911 .coefficient)
      LeftAuthority6886.bound (LeftAuthority6886.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6887RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6886.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6886.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6912 .coefficient)
      LeftAuthority6909.bound (LeftAuthority6909.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6910RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6909.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6909.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority6886.bound LeftAuthority6909.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6886.bound, LeftAuthority6909.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority6886.actual selector witness) * (LeftAuthority6909.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6913

namespace LeftBound6921
def owner : Owner := ⟨.program ⟨214⟩, ⟨18184⟩⟩
def transferEvent : Nat := 6921
def frameStart : Nat := 6825
def rule : BoundRule := .sum [.predecessor 0 6919 .coefficient, .predecessor 1 6920 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6919 .coefficient)
      LeftAuthority6917.bound (LeftAuthority6917.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6917.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6917.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6920 .coefficient)
      LeftBound6913.bound (LeftBound6913.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6913.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6913.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority6917.bound, LeftBound6913.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6917.bound, LeftBound6913.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority6917.actual selector witness, LeftBound6913.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6921

namespace LeftBound6925
def owner : Owner := ⟨.program ⟨214⟩, ⟨30213⟩⟩
def transferEvent : Nat := 6925
def frameStart : Nat := 6825
def rule : BoundRule := .sum [.predecessor 0 6923 .coefficient, .predecessor 1 6924 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6923 .coefficient)
      LeftBound6921.bound (LeftBound6921.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6922RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6921.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6921.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6924 .coefficient)
      LeftBound6902.bound (LeftBound6902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6902.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6902.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6921.bound, LeftBound6902.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6921.bound, LeftBound6902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6921.actual selector witness, LeftBound6902.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6925

namespace LeftBound6938
def owner : Owner := ⟨.program ⟨214⟩, ⟨30208⟩⟩
def transferEvent : Nat := 6938
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6936 .coefficient, .predecessor 1 6937 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6936 .coefficient)
      LeftBound6767.bound (LeftBound6767.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6935RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6767.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6767.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6937 .coefficient)
      LeftBound6750.bound (LeftBound6750.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6757RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6750.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6750.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6767.bound, LeftBound6750.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6767.bound, LeftBound6750.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6767.actual selector witness, LeftBound6750.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6938

namespace LeftBound6941
def owner : Owner := ⟨.program ⟨214⟩, ⟨30208⟩⟩
def transferEvent : Nat := 6941
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 6935 .summary, .result 6757 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6935 .summary)
      LeftBound6769.bound (LeftBound6769.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22859⟩⟩) (rawTerms := some (Proof.Events027.exact6935RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6769.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6757 .summary)
      LeftBound6752.bound (LeftBound6752.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30207⟩⟩) (rawTerms := some (Proof.Events026.exact6757RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6752.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6769.bound, LeftBound6752.bound]
def bound : CoeffClass := .finite ⟨1292539135285018636288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6769.bound, LeftBound6752.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6769.actual selector witness, LeftBound6752.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6941

namespace LeftBound6964
def owner : Owner := ⟨.program ⟨214⟩, ⟨103⟩⟩
def transferEvent : Nat := 6964
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 6963 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6963 .coefficient)
      LeftAuthority6440.bound (LeftAuthority6440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6440.derived selector witness)

def rawBound : CoeffClass := LeftAuthority6440.bound
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority6440.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound6964

namespace LeftBound6968
def owner : Owner := ⟨.program ⟨214⟩, ⟨13189⟩⟩
def transferEvent : Nat := 6968
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 6966 .coefficient) (.predecessor 1 6967 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6966 .coefficient)
      LeftAuthority73.bound (LeftAuthority73.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact74RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6967 .coefficient)
      LeftBound6447.bound (LeftBound6447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6447.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6447.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority73.bound LeftBound6447.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73.bound, LeftBound6447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority73.actual selector witness) * (LeftBound6447.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound6968

namespace LeftBound6972
def owner : Owner := ⟨.program ⟨214⟩, ⟨6789⟩⟩
def transferEvent : Nat := 6972
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 6971 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6971 .coefficient)
      LeftAuthority5869.bound (LeftAuthority5869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5869.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5869.derived selector witness)

def rawBound : CoeffClass := LeftAuthority5869.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5869.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority5869.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound6972

namespace LeftBound6976
def owner : Owner := ⟨.program ⟨214⟩, ⟨7397⟩⟩
def transferEvent : Nat := 6976
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 6974 .coefficient) (.predecessor 1 6975 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6974 .coefficient)
      LeftBound6313.bound (LeftBound6313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6975 .coefficient)
      LeftBound6972.bound (LeftBound6972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6972.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound6313.bound LeftBound6972.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6313.bound, LeftBound6972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound6313.actual selector witness) * (LeftBound6972.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6976

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
