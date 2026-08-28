import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard057
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard465

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound68704
def owner : Owner := ⟨.program ⟨214⟩, ⟨11759⟩⟩
def transferEvent : Nat := 68704
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9605⟩⟩], []⟩ [⟨.result 3250 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3250 .coefficient)
      LeftAuthority3249.bound (LeftAuthority3249.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9605⟩⟩) (rawTerms := some (Proof.Events012.exact3250RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3249.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3249.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3249.bound []
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3249.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority3249.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound68704

namespace LeftBound68705
def owner : Owner := ⟨.program ⟨214⟩, ⟨11759⟩⟩
def transferEvent : Nat := 68705
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 68700 .summary) (.transfer 68704) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68700 .summary)
      LeftBound68698.bound (LeftBound68698.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11758⟩⟩) (rawTerms := some (Proof.Events268.exact68700RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68698.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 68704)
      LeftBound68704.bound (LeftBound68704.actual selector witness) := by
  exact .transfer (LeftBound68704.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound68698.bound LeftBound68704.bound
def bound : CoeffClass := .finite ⟨24960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68698.bound, LeftBound68704.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound68698.actual selector witness) * (LeftBound68704.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68705

namespace LeftBound68711
def owner : Owner := ⟨.program ⟨214⟩, ⟨9606⟩⟩
def transferEvent : Nat := 68711
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 68709 .coefficient) (.predecessor 1 68710 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68709 .coefficient)
      LeftAuthority3249.bound (LeftAuthority3249.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3249.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3249.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68710 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3249.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3249.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3249.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound68711

namespace LeftBound68716
def owner : Owner := ⟨.program ⟨214⟩, ⟨7181⟩⟩
def transferEvent : Nat := 68716
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 68714 .coefficient) (.predecessor 1 68715 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68714 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68715 .coefficient)
      LeftBound10019.bound (LeftBound10019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10019.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound10019.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound10019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound10019.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68716

namespace LeftBound68721
def owner : Owner := ⟨.program ⟨214⟩, ⟨9607⟩⟩
def transferEvent : Nat := 68721
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 68719 .coefficient, .predecessor 1 68720 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68719 .coefficient)
      LeftBound68716.bound (LeftBound68716.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68718RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68716.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68716.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68720 .coefficient)
      LeftBound68711.bound (LeftBound68711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68711.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68711.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68716.bound, LeftBound68711.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68716.bound, LeftBound68711.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68716.actual selector witness, LeftBound68711.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68721

namespace LeftBound68725
def owner : Owner := ⟨.program ⟨214⟩, ⟨9608⟩⟩
def transferEvent : Nat := 68725
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 68723 .coefficient, .predecessor 1 68724 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68723 .coefficient)
      LeftBound68721.bound (LeftBound68721.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68722RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68721.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68721.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68724 .coefficient)
      LeftBound10011.bound (LeftBound10011.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10011.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10011.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68721.bound, LeftBound10011.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68721.bound, LeftBound10011.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68721.actual selector witness, LeftBound10011.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68725

namespace LeftBound68726
def owner : Owner := ⟨.program ⟨214⟩, ⟨9608⟩⟩
def transferEvent : Nat := 68726
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨77⟩⟩]⟩ [⟨.result 10012 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10012 .coefficient)
      LeftBound10011.bound (LeftBound10011.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨77⟩⟩) (rawTerms := some (Proof.Events039.exact10012RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10011.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10011.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound10011.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10011.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound10011.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound68726

namespace LeftBound68731
def owner : Owner := ⟨.program ⟨214⟩, ⟨9609⟩⟩
def transferEvent : Nat := 68731
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 68729 .coefficient) (.predecessor 1 68730 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68729 .coefficient)
      LeftBound68725.bound (LeftBound68725.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68725.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68725.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68730 .coefficient)
      LeftBound10008.bound (LeftBound10008.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10009RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10008.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10008.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound68725.bound LeftBound10008.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68725.bound, LeftBound10008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound68725.actual selector witness) * (LeftBound10008.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68731

namespace LeftBound68732
def owner : Owner := ⟨.program ⟨214⟩, ⟨9609⟩⟩
def transferEvent : Nat := 68732
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩ [⟨.result 10005 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10005 .coefficient)
      LeftAuthority10004.bound (LeftAuthority10004.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7861⟩⟩) (rawTerms := some (Proof.Events039.exact10005RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10004.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10004.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority10004.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10004.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10004.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound68732

namespace LeftBound68733
def owner : Owner := ⟨.program ⟨214⟩, ⟨9609⟩⟩
def transferEvent : Nat := 68733
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 68728 .summary) (.transfer 68732) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68728 .summary)
      LeftBound68726.bound (LeftBound68726.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9608⟩⟩) (rawTerms := some (Proof.Events268.exact68728RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 68732)
      LeftBound68732.bound (LeftBound68732.actual selector witness) := by
  exact .transfer (LeftBound68732.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound68726.bound LeftBound68732.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68726.bound, LeftBound68732.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound68726.actual selector witness) * (LeftBound68732.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68733

namespace LeftBound68741
def owner : Owner := ⟨.program ⟨214⟩, ⟨11760⟩⟩
def transferEvent : Nat := 68741
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 68739 .coefficient, .predecessor 1 68740 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68739 .coefficient)
      LeftBound68731.bound (LeftBound68731.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68738RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68731.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68731.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68740 .coefficient)
      LeftBound68703.bound (LeftBound68703.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68708RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68703.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68703.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68731.bound, LeftBound68703.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68731.bound, LeftBound68703.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68731.actual selector witness, LeftBound68703.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68741

namespace LeftBound68743
def owner : Owner := ⟨.program ⟨214⟩, ⟨11760⟩⟩
def transferEvent : Nat := 68743
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 68738 .summary, .result 68708 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68738 .summary)
      LeftBound68733.bound (LeftBound68733.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9609⟩⟩) (rawTerms := some (Proof.Events268.exact68738RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68733.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68708 .summary)
      LeftBound68705.bound (LeftBound68705.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11759⟩⟩) (rawTerms := some (Proof.Events268.exact68708RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68705.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68733.bound, LeftBound68705.bound]
def bound : CoeffClass := .finite ⟨95445376, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68733.bound, LeftBound68705.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68733.actual selector witness, LeftBound68705.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68743

namespace LeftBound68747
def owner : Owner := ⟨.program ⟨214⟩, ⟨25138⟩⟩
def transferEvent : Nat := 68747
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 68745 .coefficient) (.predecessor 1 68746 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68745 .coefficient)
      LeftBound68741.bound (LeftBound68741.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68741.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68741.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68746 .coefficient)
      LeftAuthority68679.bound (LeftAuthority68679.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68680RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68679.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68679.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound68741.bound LeftAuthority68679.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68741.bound, LeftAuthority68679.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound68741.actual selector witness) * (LeftAuthority68679.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68747

namespace LeftBound68748
def owner : Owner := ⟨.program ⟨214⟩, ⟨25138⟩⟩
def transferEvent : Nat := 68748
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25137⟩⟩]⟩ [⟨.result 68680 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68680 .coefficient)
      LeftAuthority68679.bound (LeftAuthority68679.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25137⟩⟩) (rawTerms := some (Proof.Events268.exact68680RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68679.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68679.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority68679.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68679.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority68679.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound68748

namespace LeftBound68749
def owner : Owner := ⟨.program ⟨214⟩, ⟨25138⟩⟩
def transferEvent : Nat := 68749
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 68744 .summary) (.transfer 68748) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68744 .summary)
      LeftBound68743.bound (LeftBound68743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11760⟩⟩) (rawTerms := some (Proof.Events268.exact68744RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 68748)
      LeftBound68748.bound (LeftBound68748.actual selector witness) := by
  exact .transfer (LeftBound68748.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound68743.bound LeftBound68748.bound
def bound : CoeffClass := .finite ⟨350286057046016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68743.bound, LeftBound68748.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound68743.actual selector witness) * (LeftBound68748.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68749

namespace LeftBound68760
def owner : Owner := ⟨.program ⟨214⟩, ⟨19742⟩⟩
def transferEvent : Nat := 68760
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 68758 .coefficient) (.value (.predecessor 1 68759 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68758 .coefficient)
      LeftAuthority68756.bound (LeftAuthority68756.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68757RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68756.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68756.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68759 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority68756.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68756.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority68756.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound68760

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
