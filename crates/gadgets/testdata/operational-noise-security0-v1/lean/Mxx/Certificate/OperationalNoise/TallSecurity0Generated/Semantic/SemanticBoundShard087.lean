import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard028
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard086

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound13781
def owner : Owner := ⟨.program ⟨214⟩, ⟨20843⟩⟩
def transferEvent : Nat := 13781
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 13779 .coefficient) (.predecessor 1 13780 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13779 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13780 .coefficient)
      LeftBound13777.bound (LeftBound13777.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13777.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13777.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound13777.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound13777.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound13777.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13781

namespace LeftBound13782
def owner : Owner := ⟨.program ⟨214⟩, ⟨20843⟩⟩
def transferEvent : Nat := 13782
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20840⟩⟩]⟩ [⟨.result 13774 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13774 .coefficient)
      LeftAuthority13773.bound (LeftAuthority13773.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20840⟩⟩) (rawTerms := some (Proof.Events053.exact13774RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13773.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13773.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority13773.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13773.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority13773.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound13782

namespace LeftBound13783
def owner : Owner := ⟨.program ⟨214⟩, ⟨20843⟩⟩
def transferEvent : Nat := 13783
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 13782) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 13782)
      LeftBound13782.bound (LeftBound13782.actual selector witness) := by
  exact .transfer (LeftBound13782.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound13782.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound13782.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound13782.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13783

namespace LeftBound13878
def owner : Owner := ⟨.program ⟨214⟩, ⟨15439⟩⟩
def transferEvent : Nat := 13878
def frameStart : Nat := 13839
def rule : BoundRule := .identity (.predecessor 0 13877 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13877 .coefficient)
      LeftAuthority13875.bound (LeftAuthority13875.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13875.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13875.derived selector witness)

def rawBound : CoeffClass := LeftAuthority13875.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13875.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority13875.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound13878

namespace LeftBound13895
def owner : Owner := ⟨.program ⟨214⟩, ⟨15478⟩⟩
def transferEvent : Nat := 13895
def frameStart : Nat := 13839
def rule : BoundRule := .sum [.predecessor 0 13893 .coefficient, .predecessor 1 13894 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13893 .coefficient)
      LeftBound13878.bound (LeftBound13878.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound13878.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13894 .coefficient)
      LeftAuthority13891.bound (LeftAuthority13891.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority13891.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13878.bound, LeftAuthority13891.bound]
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13878.bound, LeftAuthority13891.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13878.actual selector witness, LeftAuthority13891.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13895

namespace LeftBound13898
def owner : Owner := ⟨.program ⟨214⟩, ⟨15479⟩⟩
def transferEvent : Nat := 13898
def frameStart : Nat := 13839
def rule : BoundRule := .identity (.predecessor 0 13897 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13897 .coefficient)
      LeftBound13895.bound (LeftBound13895.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound13895.derived selector witness)

def rawBound : CoeffClass := LeftBound13895.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13895.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound13895.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound13898

namespace LeftBound13904
def owner : Owner := ⟨.program ⟨214⟩, ⟨15480⟩⟩
def transferEvent : Nat := 13904
def frameStart : Nat := 13839
def rule : BoundRule := .product (.predecessor 0 13902 .coefficient) (.predecessor 1 13903 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13902 .coefficient)
      LeftAuthority13900.bound (LeftAuthority13900.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13901RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13900.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13900.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13903 .coefficient)
      LeftBound13898.bound (LeftBound13898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13898.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority13900.bound LeftBound13898.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13900.bound, LeftBound13898.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority13900.actual selector witness) * (LeftBound13898.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13904

namespace LeftBound13912
def owner : Owner := ⟨.program ⟨214⟩, ⟨15481⟩⟩
def transferEvent : Nat := 13912
def frameStart : Nat := 13839
def rule : BoundRule := .sum [.predecessor 0 13910 .coefficient, .predecessor 1 13911 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13910 .coefficient)
      LeftAuthority13908.bound (LeftAuthority13908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13908.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13908.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13911 .coefficient)
      LeftBound13904.bound (LeftBound13904.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13906RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13904.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13904.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority13908.bound, LeftBound13904.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13908.bound, LeftBound13904.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority13908.actual selector witness, LeftBound13904.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13912

namespace LeftBound13916
def owner : Owner := ⟨.program ⟨214⟩, ⟨27051⟩⟩
def transferEvent : Nat := 13916
def frameStart : Nat := 13839
def rule : BoundRule := .product (.predecessor 0 13914 .coefficient) (.predecessor 1 13915 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13914 .coefficient)
      LeftBound13912.bound (LeftBound13912.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13913RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13912.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13912.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13915 .coefficient)
      LeftAuthority13889.bound (LeftAuthority13889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13890RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13889.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13889.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13912.bound LeftAuthority13889.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13912.bound, LeftAuthority13889.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13912.actual selector witness) * (LeftAuthority13889.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13916

namespace LeftBound13927
def owner : Owner := ⟨.program ⟨214⟩, ⟨17370⟩⟩
def transferEvent : Nat := 13927
def frameStart : Nat := 13839
def rule : BoundRule := .product (.predecessor 0 13925 .coefficient) (.predecessor 1 13926 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13925 .coefficient)
      LeftAuthority13900.bound (LeftAuthority13900.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13901RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13900.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13900.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13926 .coefficient)
      LeftAuthority13923.bound (LeftAuthority13923.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13923.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13923.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority13900.bound LeftAuthority13923.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13900.bound, LeftAuthority13923.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority13900.actual selector witness) * (LeftAuthority13923.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13927

namespace LeftBound13935
def owner : Owner := ⟨.program ⟨214⟩, ⟨17371⟩⟩
def transferEvent : Nat := 13935
def frameStart : Nat := 13839
def rule : BoundRule := .sum [.predecessor 0 13933 .coefficient, .predecessor 1 13934 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13933 .coefficient)
      LeftAuthority13931.bound (LeftAuthority13931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13931.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13931.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13934 .coefficient)
      LeftBound13927.bound (LeftBound13927.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13927.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13927.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority13931.bound, LeftBound13927.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13931.bound, LeftBound13927.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority13931.actual selector witness, LeftBound13927.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13935

namespace LeftBound13939
def owner : Owner := ⟨.program ⟨214⟩, ⟨27055⟩⟩
def transferEvent : Nat := 13939
def frameStart : Nat := 13839
def rule : BoundRule := .sum [.predecessor 0 13937 .coefficient, .predecessor 1 13938 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13937 .coefficient)
      LeftBound13935.bound (LeftBound13935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13936RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13935.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13935.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13938 .coefficient)
      LeftBound13916.bound (LeftBound13916.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13916.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13916.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13935.bound, LeftBound13916.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13935.bound, LeftBound13916.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13935.actual selector witness, LeftBound13916.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13939

namespace LeftBound13952
def owner : Owner := ⟨.program ⟨214⟩, ⟨27053⟩⟩
def transferEvent : Nat := 13952
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13950 .coefficient, .predecessor 1 13951 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13950 .coefficient)
      LeftBound13781.bound (LeftBound13781.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13949RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13781.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13781.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13951 .coefficient)
      LeftBound13764.bound (LeftBound13764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13771RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13764.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13764.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13781.bound, LeftBound13764.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13781.bound, LeftBound13764.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13781.actual selector witness, LeftBound13764.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13952

namespace LeftBound13955
def owner : Owner := ⟨.program ⟨214⟩, ⟨27053⟩⟩
def transferEvent : Nat := 13955
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 13949 .summary, .result 13771 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13949 .summary)
      LeftBound13783.bound (LeftBound13783.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20843⟩⟩) (rawTerms := some (Proof.Events054.exact13949RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13783.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13771 .summary)
      LeftBound13766.bound (LeftBound13766.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27052⟩⟩) (rawTerms := some (Proof.Events053.exact13771RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13766.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13783.bound, LeftBound13766.bound]
def bound : CoeffClass := .finite ⟨1291933999269462814720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13783.bound, LeftBound13766.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13783.actual selector witness, LeftBound13766.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13955

namespace LeftBound13978
def owner : Owner := ⟨.program ⟨214⟩, ⟨88⟩⟩
def transferEvent : Nat := 13978
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 13977 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13977 .coefficient)
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
end LeftBound13978

namespace LeftBound13982
def owner : Owner := ⟨.program ⟨214⟩, ⟨11012⟩⟩
def transferEvent : Nat := 13982
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 13980 .coefficient) (.predecessor 1 13981 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13980 .coefficient)
      LeftAuthority395.bound (LeftAuthority395.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events001.exact396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority395.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority395.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13981 .coefficient)
      LeftBound6447.bound (LeftBound6447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6447.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6447.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority395.bound LeftBound6447.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority395.bound, LeftBound6447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority395.actual selector witness) * (LeftBound6447.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound13982

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
