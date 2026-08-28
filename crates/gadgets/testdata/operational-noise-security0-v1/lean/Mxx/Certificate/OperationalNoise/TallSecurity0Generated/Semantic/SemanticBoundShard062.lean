import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard061

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound10572
def owner : Owner := ⟨.program ⟨214⟩, ⟨19691⟩⟩
def transferEvent : Nat := 10572
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 10570 .coefficient) (.predecessor 1 10571 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10570 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10571 .coefficient)
      LeftBound10568.bound (LeftBound10568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10569RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10568.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10568.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound10568.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound10568.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound10568.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10572

namespace LeftBound10573
def owner : Owner := ⟨.program ⟨214⟩, ⟨19691⟩⟩
def transferEvent : Nat := 10573
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19688⟩⟩]⟩ [⟨.result 10565 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10565 .coefficient)
      LeftAuthority10564.bound (LeftAuthority10564.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19688⟩⟩) (rawTerms := some (Proof.Events041.exact10565RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10564.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10564.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority10564.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10564.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10564.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound10573

namespace LeftBound10574
def owner : Owner := ⟨.program ⟨214⟩, ⟨19691⟩⟩
def transferEvent : Nat := 10574
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 10573) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 10573)
      LeftBound10573.bound (LeftBound10573.actual selector witness) := by
  exact .transfer (LeftBound10573.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound10573.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound10573.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound10573.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10574

namespace LeftBound10653
def owner : Owner := ⟨.program ⟨214⟩, ⟨14678⟩⟩
def transferEvent : Nat := 10653
def frameStart : Nat := 10624
def rule : BoundRule := .product (.predecessor 0 10651 .coefficient) (.predecessor 1 10652 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10651 .coefficient)
      LeftAuthority10649.bound (LeftAuthority10649.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10650RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10649.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10649.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10652 .coefficient)
      LeftAuthority10646.bound (LeftAuthority10646.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10647RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10646.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10646.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority10649.bound LeftAuthority10646.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10649.bound, LeftAuthority10646.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority10649.actual selector witness) * (LeftAuthority10646.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10653

namespace LeftBound10657
def owner : Owner := ⟨.program ⟨214⟩, ⟨14679⟩⟩
def transferEvent : Nat := 10657
def frameStart : Nat := 10624
def rule : BoundRule := .identity (.predecessor 0 10656 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10656 .coefficient)
      LeftBound10653.bound (LeftBound10653.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10653.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10653.derived selector witness)

def rawBound : CoeffClass := LeftBound10653.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10653.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound10653.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound10657

namespace LeftBound10674
def owner : Owner := ⟨.program ⟨214⟩, ⟨14764⟩⟩
def transferEvent : Nat := 10674
def frameStart : Nat := 10624
def rule : BoundRule := .sum [.predecessor 0 10672 .coefficient, .predecessor 1 10673 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10672 .coefficient)
      LeftBound10657.bound (LeftBound10657.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound10657.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10673 .coefficient)
      LeftAuthority10670.bound (LeftAuthority10670.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority10670.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10657.bound, LeftAuthority10670.bound]
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10657.bound, LeftAuthority10670.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound10657.actual selector witness, LeftAuthority10670.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10674

namespace LeftBound10677
def owner : Owner := ⟨.program ⟨214⟩, ⟨14765⟩⟩
def transferEvent : Nat := 10677
def frameStart : Nat := 10624
def rule : BoundRule := .identity (.predecessor 0 10676 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10676 .coefficient)
      LeftBound10674.bound (LeftBound10674.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound10674.derived selector witness)

def rawBound : CoeffClass := LeftBound10674.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound10674.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound10677

namespace LeftBound10683
def owner : Owner := ⟨.program ⟨214⟩, ⟨14766⟩⟩
def transferEvent : Nat := 10683
def frameStart : Nat := 10624
def rule : BoundRule := .product (.predecessor 0 10681 .coefficient) (.predecessor 1 10682 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10681 .coefficient)
      LeftAuthority10679.bound (LeftAuthority10679.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10680RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10679.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10679.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10682 .coefficient)
      LeftBound10677.bound (LeftBound10677.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10678RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10677.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10677.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority10679.bound LeftBound10677.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10679.bound, LeftBound10677.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority10679.actual selector witness) * (LeftBound10677.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10683

namespace LeftBound10699
def owner : Owner := ⟨.program ⟨214⟩, ⟨7859⟩⟩
def transferEvent : Nat := 10699
def frameStart : Nat := 10624
def rule : BoundRule := .scale (.predecessor 0 10697 .coefficient) (.value (.predecessor 1 10698 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10697 .coefficient)
      LeftAuthority10695.bound (LeftAuthority10695.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10696RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10695.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10695.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10698 .coefficient)
      LeftAuthority10686.bound (LeftAuthority10686.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority10686.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority10695.bound LeftAuthority10686.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10695.bound, LeftAuthority10686.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10695.actual selector witness) * (LeftAuthority10686.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound10699

namespace LeftBound10702
def owner : Owner := ⟨.program ⟨214⟩, ⟨6762⟩⟩
def transferEvent : Nat := 10702
def frameStart : Nat := 10624
def rule : BoundRule := .identity (.predecessor 0 10701 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10701 .coefficient)
      LeftAuthority10689.bound (LeftAuthority10689.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10690RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10689.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10689.derived selector witness)

def rawBound : CoeffClass := LeftAuthority10689.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10689.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority10689.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound10702

namespace LeftBound10706
def owner : Owner := ⟨.program ⟨214⟩, ⟨7860⟩⟩
def transferEvent : Nat := 10706
def frameStart : Nat := 10624
def rule : BoundRule := .product (.predecessor 0 10704 .coefficient) (.predecessor 1 10705 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10704 .coefficient)
      LeftBound10702.bound (LeftBound10702.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10703RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10702.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10702.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10705 .coefficient)
      LeftBound10699.bound (LeftBound10699.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10699.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10699.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound10702.bound LeftBound10699.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10702.bound, LeftBound10699.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound10702.actual selector witness) * (LeftBound10699.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10706

namespace LeftBound10711
def owner : Owner := ⟨.program ⟨214⟩, ⟨14767⟩⟩
def transferEvent : Nat := 10711
def frameStart : Nat := 10624
def rule : BoundRule := .sum [.predecessor 0 10709 .coefficient, .predecessor 1 10710 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10709 .coefficient)
      LeftBound10706.bound (LeftBound10706.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10708RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10706.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10706.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10710 .coefficient)
      LeftBound10683.bound (LeftBound10683.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10685RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10683.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10683.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10706.bound, LeftBound10683.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10706.bound, LeftBound10683.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound10706.actual selector witness, LeftBound10683.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10711

namespace LeftBound10715
def owner : Owner := ⟨.program ⟨214⟩, ⟨26243⟩⟩
def transferEvent : Nat := 10715
def frameStart : Nat := 10624
def rule : BoundRule := .product (.predecessor 0 10713 .coefficient) (.predecessor 1 10714 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10713 .coefficient)
      LeftBound10711.bound (LeftBound10711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10712RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10711.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10711.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10714 .coefficient)
      LeftAuthority10668.bound (LeftAuthority10668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10669RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10668.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10668.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound10711.bound LeftAuthority10668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10711.bound, LeftAuthority10668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound10711.actual selector witness) * (LeftAuthority10668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10715

namespace LeftBound10726
def owner : Owner := ⟨.program ⟨214⟩, ⟨16196⟩⟩
def transferEvent : Nat := 10726
def frameStart : Nat := 10624
def rule : BoundRule := .product (.predecessor 0 10724 .coefficient) (.predecessor 1 10725 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10724 .coefficient)
      LeftAuthority10679.bound (LeftAuthority10679.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10680RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10679.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10679.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10725 .coefficient)
      LeftAuthority10722.bound (LeftAuthority10722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10722.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10722.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority10679.bound LeftAuthority10722.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10679.bound, LeftAuthority10722.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority10679.actual selector witness) * (LeftAuthority10722.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10726

namespace LeftBound10734
def owner : Owner := ⟨.program ⟨214⟩, ⟨16197⟩⟩
def transferEvent : Nat := 10734
def frameStart : Nat := 10624
def rule : BoundRule := .sum [.predecessor 0 10732 .coefficient, .predecessor 1 10733 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10732 .coefficient)
      LeftAuthority10730.bound (LeftAuthority10730.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10731RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10730.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10730.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10733 .coefficient)
      LeftBound10726.bound (LeftBound10726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10726.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority10730.bound, LeftBound10726.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10730.bound, LeftBound10726.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority10730.actual selector witness, LeftBound10726.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10734

namespace LeftBound10738
def owner : Owner := ⟨.program ⟨214⟩, ⟨26244⟩⟩
def transferEvent : Nat := 10738
def frameStart : Nat := 10624
def rule : BoundRule := .sum [.predecessor 0 10736 .coefficient, .predecessor 1 10737 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10736 .coefficient)
      LeftBound10734.bound (LeftBound10734.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10735RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10734.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10734.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10737 .coefficient)
      LeftBound10715.bound (LeftBound10715.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10720RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10715.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10715.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10734.bound, LeftBound10715.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10734.bound, LeftBound10715.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound10734.actual selector witness, LeftBound10715.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10738

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
