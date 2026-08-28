import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard682
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard724

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound105816
def owner : Owner := ⟨.program ⟨214⟩, ⟨17158⟩⟩
def transferEvent : Nat := 105816
def frameStart : Nat := 105732
def rule : BoundRule := .sum [.predecessor 0 105814 .coefficient, .predecessor 1 105815 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105814 .coefficient)
      LeftAuthority105812.bound (LeftAuthority105812.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105813RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105812.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105812.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105815 .coefficient)
      LeftBound105808.bound (LeftBound105808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105810RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105808.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105808.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority105812.bound, LeftBound105808.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105812.bound, LeftBound105808.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority105812.actual selector witness, LeftBound105808.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105816

namespace LeftBound105820
def owner : Owner := ⟨.program ⟨214⟩, ⟨27830⟩⟩
def transferEvent : Nat := 105820
def frameStart : Nat := 105732
def rule : BoundRule := .sum [.predecessor 0 105818 .coefficient, .predecessor 1 105819 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105818 .coefficient)
      LeftBound105816.bound (LeftBound105816.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105817RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105816.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105816.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105819 .coefficient)
      LeftBound105797.bound (LeftBound105797.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105802RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105797.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105797.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105816.bound, LeftBound105797.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105816.bound, LeftBound105797.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound105816.actual selector witness, LeftBound105797.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105820

namespace LeftBound105833
def owner : Owner := ⟨.program ⟨214⟩, ⟨27827⟩⟩
def transferEvent : Nat := 105833
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 105831 .coefficient, .predecessor 1 105832 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105831 .coefficient)
      LeftBound105686.bound (LeftBound105686.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105830RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105686.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105686.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105832 .coefficient)
      LeftBound105669.bound (LeftBound105669.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105676RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105669.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105669.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105686.bound, LeftBound105669.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105686.bound, LeftBound105669.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound105686.actual selector witness, LeftBound105669.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105833

namespace LeftBound105836
def owner : Owner := ⟨.program ⟨214⟩, ⟨27827⟩⟩
def transferEvent : Nat := 105836
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 105830 .summary, .result 105676 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105830 .summary)
      LeftBound105688.bound (LeftBound105688.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21320⟩⟩) (rawTerms := some (Proof.Events413.exact105830RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105688.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105676 .summary)
      LeftBound105671.bound (LeftBound105671.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27826⟩⟩) (rawTerms := some (Proof.Events412.exact105676RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105671.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105688.bound, LeftBound105671.bound]
def bound : CoeffClass := .finite ⟨1292068473939586330624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105688.bound, LeftBound105671.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound105688.actual selector witness, LeftBound105671.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105836

namespace LeftBound105840
def owner : Owner := ⟨.program ⟨214⟩, ⟨27828⟩⟩
def transferEvent : Nat := 105840
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 105838 .coefficient) (.predecessor 1 105839 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105838 .coefficient)
      LeftBound105833.bound (LeftBound105833.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105837RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105833.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105833.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105839 .coefficient)
      LeftBound5718.bound (LeftBound5718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5718.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5718.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound105833.bound LeftBound5718.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105833.bound, LeftBound5718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound105833.actual selector witness) * (LeftBound5718.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105840

namespace LeftBound105841
def owner : Owner := ⟨.program ⟨214⟩, ⟨27828⟩⟩
def transferEvent : Nat := 105841
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩ [⟨.result 5715 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5715 .coefficient)
      LeftAuthority5714.bound (LeftAuthority5714.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6641⟩⟩) (rawTerms := some (Proof.Events022.exact5715RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5714.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5714.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5714.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5714.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5714.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound105841

namespace LeftBound105842
def owner : Owner := ⟨.program ⟨214⟩, ⟨27828⟩⟩
def transferEvent : Nat := 105842
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 105837 .summary) (.transfer 105841) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105837 .summary)
      LeftBound105836.bound (LeftBound105836.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27827⟩⟩) (rawTerms := some (Proof.Events413.exact105837RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105836.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 105841)
      LeftBound105841.bound (LeftBound105841.actual selector witness) := by
  exact .transfer (LeftBound105841.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound105836.bound LeftBound105841.bound
def bound : CoeffClass := .finite ⟨4741911972453864866771369984, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105836.bound, LeftBound105841.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound105836.actual selector witness) * (LeftBound105841.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105842

namespace LeftBound105857
def owner : Owner := ⟨.program ⟨214⟩, ⟨27609⟩⟩
def transferEvent : Nat := 105857
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 105855 .coefficient) (.predecessor 1 105856 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105855 .coefficient)
      LeftBound99394.bound (LeftBound99394.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99394.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99394.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105856 .coefficient)
      LeftAuthority105853.bound (LeftAuthority105853.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105854RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105853.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105853.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99394.bound LeftAuthority105853.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99394.bound, LeftAuthority105853.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99394.actual selector witness) * (LeftAuthority105853.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105857

namespace LeftBound105858
def owner : Owner := ⟨.program ⟨214⟩, ⟨27609⟩⟩
def transferEvent : Nat := 105858
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27607⟩⟩]⟩ [⟨.result 105854 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105854 .coefficient)
      LeftAuthority105853.bound (LeftAuthority105853.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27607⟩⟩) (rawTerms := some (Proof.Events413.exact105854RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105853.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105853.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority105853.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105853.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority105853.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound105858

namespace LeftBound105859
def owner : Owner := ⟨.program ⟨214⟩, ⟨27609⟩⟩
def transferEvent : Nat := 105859
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 99398 .summary) (.transfer 105858) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99398 .summary)
      LeftBound99397.bound (LeftBound99397.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25978⟩⟩) (rawTerms := some (Proof.Events388.exact99398RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 105858)
      LeftBound105858.bound (LeftBound105858.actual selector witness) := by
  exact .transfer (LeftBound105858.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99397.bound LeftBound105858.bound
def bound : CoeffClass := .finite ⟨1292046059683262234624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99397.bound, LeftBound105858.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99397.actual selector witness) * (LeftBound105858.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105859

namespace LeftBound105870
def owner : Owner := ⟨.program ⟨214⟩, ⟨21175⟩⟩
def transferEvent : Nat := 105870
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 105868 .coefficient) (.value (.predecessor 1 105869 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105868 .coefficient)
      LeftAuthority105866.bound (LeftAuthority105866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105867RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105866.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105866.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105869 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority105866.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105866.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority105866.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound105870

namespace LeftBound105874
def owner : Owner := ⟨.program ⟨214⟩, ⟨21176⟩⟩
def transferEvent : Nat := 105874
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 105872 .coefficient) (.predecessor 1 105873 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105872 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105873 .coefficient)
      LeftBound105870.bound (LeftBound105870.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105871RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105870.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105870.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound105870.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound105870.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound105870.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105874

namespace LeftBound105875
def owner : Owner := ⟨.program ⟨214⟩, ⟨21176⟩⟩
def transferEvent : Nat := 105875
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21173⟩⟩]⟩ [⟨.result 105867 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105867 .coefficient)
      LeftAuthority105866.bound (LeftAuthority105866.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21173⟩⟩) (rawTerms := some (Proof.Events413.exact105867RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105866.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105866.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority105866.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority105866.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound105875

namespace LeftBound105876
def owner : Owner := ⟨.program ⟨214⟩, ⟨21176⟩⟩
def transferEvent : Nat := 105876
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 105875) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 105875)
      LeftBound105875.bound (LeftBound105875.actual selector witness) := by
  exact .transfer (LeftBound105875.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound105875.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound105875.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound105875.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105876

namespace LeftBound105947
def owner : Owner := ⟨.program ⟨214⟩, ⟨15812⟩⟩
def transferEvent : Nat := 105947
def frameStart : Nat := 105920
def rule : BoundRule := .identity (.predecessor 0 105946 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105946 .coefficient)
      LeftAuthority105944.bound (LeftAuthority105944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105944.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105944.derived selector witness)

def rawBound : CoeffClass := LeftAuthority105944.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105944.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority105944.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound105947

namespace LeftBound105964
def owner : Owner := ⟨.program ⟨214⟩, ⟨15888⟩⟩
def transferEvent : Nat := 105964
def frameStart : Nat := 105920
def rule : BoundRule := .sum [.predecessor 0 105962 .coefficient, .predecessor 1 105963 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105962 .coefficient)
      LeftBound105947.bound (LeftBound105947.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound105947.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105963 .coefficient)
      LeftAuthority105960.bound (LeftAuthority105960.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority105960.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105947.bound, LeftAuthority105960.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105947.bound, LeftAuthority105960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound105947.actual selector witness, LeftAuthority105960.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105964

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
