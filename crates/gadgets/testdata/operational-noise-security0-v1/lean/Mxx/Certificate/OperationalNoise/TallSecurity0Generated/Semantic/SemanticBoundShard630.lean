import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard594
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard629

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound93061
def owner : Owner := ⟨.program ⟨214⟩, ⟨15584⟩⟩
def transferEvent : Nat := 93061
def frameStart : Nat := 93022
def rule : BoundRule := .identity (.predecessor 0 93060 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93060 .coefficient)
      LeftAuthority93058.bound (LeftAuthority93058.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact93059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93058.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93058.derived selector witness)

def rawBound : CoeffClass := LeftAuthority93058.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93058.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority93058.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound93061

namespace LeftBound93078
def owner : Owner := ⟨.program ⟨214⟩, ⟨15658⟩⟩
def transferEvent : Nat := 93078
def frameStart : Nat := 93022
def rule : BoundRule := .sum [.predecessor 0 93076 .coefficient, .predecessor 1 93077 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93076 .coefficient)
      LeftBound93061.bound (LeftBound93061.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound93061.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93077 .coefficient)
      LeftAuthority93074.bound (LeftAuthority93074.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority93074.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound93061.bound, LeftAuthority93074.bound]
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93061.bound, LeftAuthority93074.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound93061.actual selector witness, LeftAuthority93074.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93078

namespace LeftBound93081
def owner : Owner := ⟨.program ⟨214⟩, ⟨15659⟩⟩
def transferEvent : Nat := 93081
def frameStart : Nat := 93022
def rule : BoundRule := .identity (.predecessor 0 93080 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93080 .coefficient)
      LeftBound93078.bound (LeftBound93078.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound93078.derived selector witness)

def rawBound : CoeffClass := LeftBound93078.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93078.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound93078.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound93081

namespace LeftBound93087
def owner : Owner := ⟨.program ⟨214⟩, ⟨15660⟩⟩
def transferEvent : Nat := 93087
def frameStart : Nat := 93022
def rule : BoundRule := .product (.predecessor 0 93085 .coefficient) (.predecessor 1 93086 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93085 .coefficient)
      LeftAuthority93083.bound (LeftAuthority93083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact93084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93083.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93083.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93086 .coefficient)
      LeftBound93081.bound (LeftBound93081.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact93082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93081.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93081.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority93083.bound LeftBound93081.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93083.bound, LeftBound93081.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority93083.actual selector witness) * (LeftBound93081.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93087

namespace LeftBound93095
def owner : Owner := ⟨.program ⟨214⟩, ⟨15661⟩⟩
def transferEvent : Nat := 93095
def frameStart : Nat := 93022
def rule : BoundRule := .sum [.predecessor 0 93093 .coefficient, .predecessor 1 93094 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93093 .coefficient)
      LeftAuthority93091.bound (LeftAuthority93091.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact93092RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93091.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93091.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93094 .coefficient)
      LeftBound93087.bound (LeftBound93087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact93089RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93087.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority93091.bound, LeftBound93087.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93091.bound, LeftBound93087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority93091.actual selector witness, LeftBound93087.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93095

namespace LeftBound93099
def owner : Owner := ⟨.program ⟨214⟩, ⟨27209⟩⟩
def transferEvent : Nat := 93099
def frameStart : Nat := 93022
def rule : BoundRule := .product (.predecessor 0 93097 .coefficient) (.predecessor 1 93098 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93097 .coefficient)
      LeftBound93095.bound (LeftBound93095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact93096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93095.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93095.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93098 .coefficient)
      LeftAuthority93072.bound (LeftAuthority93072.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact93073RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93072.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93072.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound93095.bound LeftAuthority93072.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93095.bound, LeftAuthority93072.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound93095.actual selector witness) * (LeftAuthority93072.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93099

namespace LeftBound93110
def owner : Owner := ⟨.program ⟨214⟩, ⟨17820⟩⟩
def transferEvent : Nat := 93110
def frameStart : Nat := 93022
def rule : BoundRule := .product (.predecessor 0 93108 .coefficient) (.predecessor 1 93109 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93108 .coefficient)
      LeftAuthority93083.bound (LeftAuthority93083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact93084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93083.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93083.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93109 .coefficient)
      LeftAuthority93106.bound (LeftAuthority93106.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact93107RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93106.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93106.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority93083.bound LeftAuthority93106.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93083.bound, LeftAuthority93106.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority93083.actual selector witness) * (LeftAuthority93106.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93110

namespace LeftBound93118
def owner : Owner := ⟨.program ⟨214⟩, ⟨17821⟩⟩
def transferEvent : Nat := 93118
def frameStart : Nat := 93022
def rule : BoundRule := .sum [.predecessor 0 93116 .coefficient, .predecessor 1 93117 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93116 .coefficient)
      LeftAuthority93114.bound (LeftAuthority93114.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact93115RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93114.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93114.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93117 .coefficient)
      LeftBound93110.bound (LeftBound93110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact93112RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93110.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93110.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority93114.bound, LeftBound93110.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93114.bound, LeftBound93110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority93114.actual selector witness, LeftBound93110.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93118

namespace LeftBound93122
def owner : Owner := ⟨.program ⟨214⟩, ⟨27214⟩⟩
def transferEvent : Nat := 93122
def frameStart : Nat := 93022
def rule : BoundRule := .sum [.predecessor 0 93120 .coefficient, .predecessor 1 93121 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93120 .coefficient)
      LeftBound93118.bound (LeftBound93118.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact93119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93118.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93121 .coefficient)
      LeftBound93099.bound (LeftBound93099.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact93104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93099.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93099.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound93118.bound, LeftBound93099.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93118.bound, LeftBound93099.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound93118.actual selector witness, LeftBound93099.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93122

namespace LeftBound93135
def owner : Owner := ⟨.program ⟨214⟩, ⟨27211⟩⟩
def transferEvent : Nat := 93135
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 93133 .coefficient, .predecessor 1 93134 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93133 .coefficient)
      LeftBound92964.bound (LeftBound92964.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact93132RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92964.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92964.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93134 .coefficient)
      LeftBound92947.bound (LeftBound92947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact92954RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92947.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92947.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92964.bound, LeftBound92947.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92964.bound, LeftBound92947.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound92964.actual selector witness, LeftBound92947.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93135

namespace LeftBound93138
def owner : Owner := ⟨.program ⟨214⟩, ⟨27211⟩⟩
def transferEvent : Nat := 93138
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 93132 .summary, .result 92954 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93132 .summary)
      LeftBound92966.bound (LeftBound92966.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20899⟩⟩) (rawTerms := some (Proof.Events363.exact93132RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92966.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92954 .summary)
      LeftBound92949.bound (LeftBound92949.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27210⟩⟩) (rawTerms := some (Proof.Events363.exact92954RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92949.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92966.bound, LeftBound92949.bound]
def bound : CoeffClass := .finite ⟨1291978824159503986688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92966.bound, LeftBound92949.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound92966.actual selector witness, LeftBound92949.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93138

namespace LeftBound93142
def owner : Owner := ⟨.program ⟨214⟩, ⟨27212⟩⟩
def transferEvent : Nat := 93142
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 93140 .coefficient) (.predecessor 1 93141 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93140 .coefficient)
      LeftBound93135.bound (LeftBound93135.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact93139RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93135.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93141 .coefficient)
      LeftBound5778.bound (LeftBound5778.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5779RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5778.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5778.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound93135.bound LeftBound5778.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93135.bound, LeftBound5778.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound93135.actual selector witness) * (LeftBound5778.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93142

namespace LeftBound93143
def owner : Owner := ⟨.program ⟨214⟩, ⟨27212⟩⟩
def transferEvent : Nat := 93143
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩ [⟨.result 5775 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5775 .coefficient)
      LeftAuthority5774.bound (LeftAuthority5774.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6649⟩⟩) (rawTerms := some (Proof.Events022.exact5775RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5774.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5774.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5774.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5774.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5774.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound93143

namespace LeftBound93144
def owner : Owner := ⟨.program ⟨214⟩, ⟨27212⟩⟩
def transferEvent : Nat := 93144
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 93139 .summary) (.transfer 93143) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93139 .summary)
      LeftBound93138.bound (LeftBound93138.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27211⟩⟩) (rawTerms := some (Proof.Events363.exact93139RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound93138.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 93143)
      LeftBound93143.bound (LeftBound93143.actual selector witness) := by
  exact .transfer (LeftBound93143.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound93138.bound LeftBound93143.bound
def bound : CoeffClass := .finite ⟨4741582956326566183208747008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93138.bound, LeftBound93143.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound93138.actual selector witness) * (LeftBound93143.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93144

namespace LeftBound93159
def owner : Owner := ⟨.program ⟨214⟩, ⟨26993⟩⟩
def transferEvent : Nat := 93159
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 93157 .coefficient) (.predecessor 1 93158 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93157 .coefficient)
      LeftBound86912.bound (LeftBound86912.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86916RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86912.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86912.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93158 .coefficient)
      LeftAuthority93155.bound (LeftAuthority93155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact93156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93155.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93155.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound86912.bound LeftAuthority93155.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86912.bound, LeftAuthority93155.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound86912.actual selector witness) * (LeftAuthority93155.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93159

namespace LeftBound93160
def owner : Owner := ⟨.program ⟨214⟩, ⟨26993⟩⟩
def transferEvent : Nat := 93160
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26991⟩⟩]⟩ [⟨.result 93156 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93156 .coefficient)
      LeftAuthority93155.bound (LeftAuthority93155.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26991⟩⟩) (rawTerms := some (Proof.Events363.exact93156RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93155.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93155.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority93155.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93155.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority93155.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound93160

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
