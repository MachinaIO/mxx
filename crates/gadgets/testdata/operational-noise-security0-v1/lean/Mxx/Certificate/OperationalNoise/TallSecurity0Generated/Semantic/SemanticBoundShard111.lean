import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard051
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard110

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound18078
def owner : Owner := ⟨.program ⟨214⟩, ⟨16605⟩⟩
def transferEvent : Nat := 18078
def frameStart : Nat := 18022
def rule : BoundRule := .sum [.predecessor 0 18076 .coefficient, .predecessor 1 18077 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18076 .coefficient)
      LeftBound18061.bound (LeftBound18061.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound18061.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18077 .coefficient)
      LeftAuthority18074.bound (LeftAuthority18074.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority18074.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound18061.bound, LeftAuthority18074.bound]
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18061.bound, LeftAuthority18074.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound18061.actual selector witness, LeftAuthority18074.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18078

namespace LeftBound18081
def owner : Owner := ⟨.program ⟨214⟩, ⟨16606⟩⟩
def transferEvent : Nat := 18081
def frameStart : Nat := 18022
def rule : BoundRule := .identity (.predecessor 0 18080 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18080 .coefficient)
      LeftBound18078.bound (LeftBound18078.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound18078.derived selector witness)

def rawBound : CoeffClass := LeftBound18078.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18078.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound18078.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound18081

namespace LeftBound18087
def owner : Owner := ⟨.program ⟨214⟩, ⟨16607⟩⟩
def transferEvent : Nat := 18087
def frameStart : Nat := 18022
def rule : BoundRule := .product (.predecessor 0 18085 .coefficient) (.predecessor 1 18086 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18085 .coefficient)
      LeftAuthority18083.bound (LeftAuthority18083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18083.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18083.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18086 .coefficient)
      LeftBound18081.bound (LeftBound18081.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18081.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18081.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority18083.bound LeftBound18081.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18083.bound, LeftBound18081.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority18083.actual selector witness) * (LeftBound18081.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18087

namespace LeftBound18095
def owner : Owner := ⟨.program ⟨214⟩, ⟨16608⟩⟩
def transferEvent : Nat := 18095
def frameStart : Nat := 18022
def rule : BoundRule := .sum [.predecessor 0 18093 .coefficient, .predecessor 1 18094 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18093 .coefficient)
      LeftAuthority18091.bound (LeftAuthority18091.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18092RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18091.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18091.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18094 .coefficient)
      LeftBound18087.bound (LeftBound18087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18089RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18087.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority18091.bound, LeftBound18087.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18091.bound, LeftBound18087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority18091.actual selector witness, LeftBound18087.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18095

namespace LeftBound18099
def owner : Owner := ⟨.program ⟨214⟩, ⟨29214⟩⟩
def transferEvent : Nat := 18099
def frameStart : Nat := 18022
def rule : BoundRule := .product (.predecessor 0 18097 .coefficient) (.predecessor 1 18098 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18097 .coefficient)
      LeftBound18095.bound (LeftBound18095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18095.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18095.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18098 .coefficient)
      LeftAuthority18072.bound (LeftAuthority18072.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18073RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18072.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18072.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound18095.bound LeftAuthority18072.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18095.bound, LeftAuthority18072.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound18095.actual selector witness) * (LeftAuthority18072.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18099

namespace LeftBound18110
def owner : Owner := ⟨.program ⟨214⟩, ⟨17967⟩⟩
def transferEvent : Nat := 18110
def frameStart : Nat := 18022
def rule : BoundRule := .product (.predecessor 0 18108 .coefficient) (.predecessor 1 18109 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18108 .coefficient)
      LeftAuthority18083.bound (LeftAuthority18083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18083.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18083.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18109 .coefficient)
      LeftAuthority18106.bound (LeftAuthority18106.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18107RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18106.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18106.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority18083.bound LeftAuthority18106.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18083.bound, LeftAuthority18106.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority18083.actual selector witness) * (LeftAuthority18106.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18110

namespace LeftBound18118
def owner : Owner := ⟨.program ⟨214⟩, ⟨17968⟩⟩
def transferEvent : Nat := 18118
def frameStart : Nat := 18022
def rule : BoundRule := .sum [.predecessor 0 18116 .coefficient, .predecessor 1 18117 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18116 .coefficient)
      LeftAuthority18114.bound (LeftAuthority18114.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18115RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18114.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18114.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18117 .coefficient)
      LeftBound18110.bound (LeftBound18110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18112RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18110.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18110.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority18114.bound, LeftBound18110.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18114.bound, LeftBound18110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority18114.actual selector witness, LeftBound18110.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18118

namespace LeftBound18122
def owner : Owner := ⟨.program ⟨214⟩, ⟨29219⟩⟩
def transferEvent : Nat := 18122
def frameStart : Nat := 18022
def rule : BoundRule := .sum [.predecessor 0 18120 .coefficient, .predecessor 1 18121 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18120 .coefficient)
      LeftBound18118.bound (LeftBound18118.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18118.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18121 .coefficient)
      LeftBound18099.bound (LeftBound18099.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18099.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18099.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound18118.bound, LeftBound18099.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18118.bound, LeftBound18099.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound18118.actual selector witness, LeftBound18099.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18122

namespace LeftBound18135
def owner : Owner := ⟨.program ⟨214⟩, ⟨29216⟩⟩
def transferEvent : Nat := 18135
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 18133 .coefficient, .predecessor 1 18134 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18133 .coefficient)
      LeftBound17964.bound (LeftBound17964.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18132RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17964.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17964.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18134 .coefficient)
      LeftBound17947.bound (LeftBound17947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact17954RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17947.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17947.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound17964.bound, LeftBound17947.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17964.bound, LeftBound17947.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound17964.actual selector witness, LeftBound17947.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18135

namespace LeftBound18138
def owner : Owner := ⟨.program ⟨214⟩, ⟨29216⟩⟩
def transferEvent : Nat := 18138
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 18132 .summary, .result 17954 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18132 .summary)
      LeftBound17966.bound (LeftBound17966.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22211⟩⟩) (rawTerms := some (Proof.Events070.exact18132RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17966.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17954 .summary)
      LeftBound17949.bound (LeftBound17949.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29215⟩⟩) (rawTerms := some (Proof.Events070.exact17954RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17949.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound17966.bound, LeftBound17949.bound]
def bound : CoeffClass := .finite ⟨1292337423279833362432, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17966.bound, LeftBound17949.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound17966.actual selector witness, LeftBound17949.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18138

namespace LeftBound18142
def owner : Owner := ⟨.program ⟨214⟩, ⟨29217⟩⟩
def transferEvent : Nat := 18142
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 18140 .coefficient) (.predecessor 1 18141 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18140 .coefficient)
      LeftBound18135.bound (LeftBound18135.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18139RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18135.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18141 .coefficient)
      LeftBound5598.bound (LeftBound5598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5599RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5598.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5598.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound18135.bound LeftBound5598.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18135.bound, LeftBound5598.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound18135.actual selector witness) * (LeftBound5598.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18142

namespace LeftBound18143
def owner : Owner := ⟨.program ⟨214⟩, ⟨29217⟩⟩
def transferEvent : Nat := 18143
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6667⟩⟩]⟩ [⟨.result 5595 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5595 .coefficient)
      LeftAuthority5594.bound (LeftAuthority5594.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6667⟩⟩) (rawTerms := some (Proof.Events021.exact5595RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5594.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5594.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5594.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5594.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5594.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound18143

namespace LeftBound18144
def owner : Owner := ⟨.program ⟨214⟩, ⟨29217⟩⟩
def transferEvent : Nat := 18144
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 18139 .summary) (.transfer 18143) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18139 .summary)
      LeftBound18138.bound (LeftBound18138.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29216⟩⟩) (rawTerms := some (Proof.Events070.exact18139RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound18138.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 18143)
      LeftBound18143.bound (LeftBound18143.actual selector witness) := by
  exact .transfer (LeftBound18143.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound18138.bound LeftBound18143.bound
def bound : CoeffClass := .finite ⟨4742899020835760917459238912, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18138.bound, LeftBound18143.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound18138.actual selector witness) * (LeftBound18143.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18144

namespace LeftBound18159
def owner : Owner := ⟨.program ⟨214⟩, ⟨28998⟩⟩
def transferEvent : Nat := 18159
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 18157 .coefficient) (.predecessor 1 18158 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18157 .coefficient)
      LeftBound9248.bound (LeftBound9248.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9252RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9248.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9248.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18158 .coefficient)
      LeftAuthority18155.bound (LeftAuthority18155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18155.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18155.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound9248.bound LeftAuthority18155.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9248.bound, LeftAuthority18155.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound9248.actual selector witness) * (LeftAuthority18155.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18159

namespace LeftBound18160
def owner : Owner := ⟨.program ⟨214⟩, ⟨28998⟩⟩
def transferEvent : Nat := 18160
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28996⟩⟩]⟩ [⟨.result 18156 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18156 .coefficient)
      LeftAuthority18155.bound (LeftAuthority18155.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28996⟩⟩) (rawTerms := some (Proof.Events070.exact18156RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18155.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18155.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority18155.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18155.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority18155.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound18160

namespace LeftBound18161
def owner : Owner := ⟨.program ⟨214⟩, ⟨28998⟩⟩
def transferEvent : Nat := 18161
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 9252 .summary) (.transfer 18160) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9252 .summary)
      LeftBound9251.bound (LeftBound9251.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25395⟩⟩) (rawTerms := some (Proof.Events036.exact9252RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound9251.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 18160)
      LeftBound18160.bound (LeftBound18160.actual selector witness) := by
  exact .transfer (LeftBound18160.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound9251.bound LeftBound18160.bound
def bound : CoeffClass := .finite ⟨1292315009023509266432, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9251.bound, LeftBound18160.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound9251.actual selector witness) * (LeftBound18160.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18161

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
