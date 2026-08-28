import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard029
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard338

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound50694
def owner : Owner := ⟨.program ⟨214⟩, ⟨13364⟩⟩
def transferEvent : Nat := 50694
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨10350⟩⟩], []⟩ [⟨.result 2341 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2341 .coefficient)
      LeftAuthority2340.bound (LeftAuthority2340.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨10350⟩⟩) (rawTerms := some (Proof.Events009.exact2341RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2340.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2340.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority2340.bound []
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2340.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority2340.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound50694

namespace LeftBound50695
def owner : Owner := ⟨.program ⟨214⟩, ⟨13364⟩⟩
def transferEvent : Nat := 50695
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50690 .summary) (.transfer 50694) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50690 .summary)
      LeftBound50688.bound (LeftBound50688.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13363⟩⟩) (rawTerms := some (Proof.Events198.exact50690RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50688.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50694)
      LeftBound50694.bound (LeftBound50694.actual selector witness) := by
  exact .transfer (LeftBound50694.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50688.bound LeftBound50694.bound
def bound : CoeffClass := .finite ⟨49920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50688.bound, LeftBound50694.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50688.actual selector witness) * (LeftBound50694.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50695

namespace LeftBound50701
def owner : Owner := ⟨.program ⟨214⟩, ⟨10351⟩⟩
def transferEvent : Nat := 50701
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 50699 .coefficient) (.predecessor 1 50700 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50699 .coefficient)
      LeftAuthority2340.bound (LeftAuthority2340.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2341RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2340.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2340.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50700 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2340.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2340.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2340.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound50701

namespace LeftBound50706
def owner : Owner := ⟨.program ⟨214⟩, ⟨7264⟩⟩
def transferEvent : Nat := 50706
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 50704 .coefficient) (.predecessor 1 50705 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50704 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50705 .coefficient)
      LeftBound6497.bound (LeftBound6497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6497.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6497.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound6497.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound6497.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound6497.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50706

namespace LeftBound50711
def owner : Owner := ⟨.program ⟨214⟩, ⟨10352⟩⟩
def transferEvent : Nat := 50711
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50709 .coefficient, .predecessor 1 50710 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50709 .coefficient)
      LeftBound50706.bound (LeftBound50706.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50708RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50706.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50706.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50710 .coefficient)
      LeftBound50701.bound (LeftBound50701.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50703RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50701.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50701.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50706.bound, LeftBound50701.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50706.bound, LeftBound50701.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50706.actual selector witness, LeftBound50701.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50711

namespace LeftBound50715
def owner : Owner := ⟨.program ⟨214⟩, ⟨10353⟩⟩
def transferEvent : Nat := 50715
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50713 .coefficient, .predecessor 1 50714 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50713 .coefficient)
      LeftBound50711.bound (LeftBound50711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50712RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50711.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50711.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50714 .coefficient)
      LeftBound6489.bound (LeftBound6489.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6489.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6489.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50711.bound, LeftBound6489.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50711.bound, LeftBound6489.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50711.actual selector witness, LeftBound6489.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50715

namespace LeftBound50716
def owner : Owner := ⟨.program ⟨214⟩, ⟨10353⟩⟩
def transferEvent : Nat := 50716
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨84⟩⟩]⟩ [⟨.result 6490 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6490 .coefficient)
      LeftBound6489.bound (LeftBound6489.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨84⟩⟩) (rawTerms := some (Proof.Events025.exact6490RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6489.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6489.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound6489.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6489.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound6489.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound50716

namespace LeftBound50721
def owner : Owner := ⟨.program ⟨214⟩, ⟨10354⟩⟩
def transferEvent : Nat := 50721
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 50719 .coefficient) (.predecessor 1 50720 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50719 .coefficient)
      LeftBound50715.bound (LeftBound50715.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50718RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50715.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50715.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50720 .coefficient)
      LeftBound6486.bound (LeftBound6486.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6487RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6486.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6486.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound50715.bound LeftBound6486.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50715.bound, LeftBound6486.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound50715.actual selector witness) * (LeftBound6486.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50721

namespace LeftBound50722
def owner : Owner := ⟨.program ⟨214⟩, ⟨10354⟩⟩
def transferEvent : Nat := 50722
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩ [⟨.result 6483 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6483 .coefficient)
      LeftAuthority6482.bound (LeftAuthority6482.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7882⟩⟩) (rawTerms := some (Proof.Events025.exact6483RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6482.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6482.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6482.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6482.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6482.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound50722

namespace LeftBound50723
def owner : Owner := ⟨.program ⟨214⟩, ⟨10354⟩⟩
def transferEvent : Nat := 50723
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50718 .summary) (.transfer 50722) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50718 .summary)
      LeftBound50716.bound (LeftBound50716.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10353⟩⟩) (rawTerms := some (Proof.Events198.exact50718RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50716.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50722)
      LeftBound50722.bound (LeftBound50722.actual selector witness) := by
  exact .transfer (LeftBound50722.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound50716.bound LeftBound50722.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50716.bound, LeftBound50722.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound50716.actual selector witness) * (LeftBound50722.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50723

namespace LeftBound50731
def owner : Owner := ⟨.program ⟨214⟩, ⟨13365⟩⟩
def transferEvent : Nat := 50731
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50729 .coefficient, .predecessor 1 50730 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50729 .coefficient)
      LeftBound50721.bound (LeftBound50721.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50721.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50721.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50730 .coefficient)
      LeftBound50693.bound (LeftBound50693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50698RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50693.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50693.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50721.bound, LeftBound50693.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50721.bound, LeftBound50693.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50721.actual selector witness, LeftBound50693.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50731

namespace LeftBound50733
def owner : Owner := ⟨.program ⟨214⟩, ⟨13365⟩⟩
def transferEvent : Nat := 50733
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50728 .summary, .result 50698 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50728 .summary)
      LeftBound50723.bound (LeftBound50723.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10354⟩⟩) (rawTerms := some (Proof.Events198.exact50728RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50723.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50698 .summary)
      LeftBound50695.bound (LeftBound50695.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13364⟩⟩) (rawTerms := some (Proof.Events198.exact50698RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50695.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50723.bound, LeftBound50695.bound]
def bound : CoeffClass := .finite ⟨95470336, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50723.bound, LeftBound50695.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50723.actual selector witness, LeftBound50695.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50733

namespace LeftBound50737
def owner : Owner := ⟨.program ⟨214⟩, ⟨25764⟩⟩
def transferEvent : Nat := 50737
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 50735 .coefficient) (.predecessor 1 50736 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50735 .coefficient)
      LeftBound50731.bound (LeftBound50731.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50734RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50731.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50731.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50736 .coefficient)
      LeftAuthority50664.bound (LeftAuthority50664.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50664.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50664.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound50731.bound LeftAuthority50664.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50731.bound, LeftAuthority50664.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound50731.actual selector witness) * (LeftAuthority50664.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50737

namespace LeftBound50738
def owner : Owner := ⟨.program ⟨214⟩, ⟨25764⟩⟩
def transferEvent : Nat := 50738
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25763⟩⟩]⟩ [⟨.result 50665 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50665 .coefficient)
      LeftAuthority50664.bound (LeftAuthority50664.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25763⟩⟩) (rawTerms := some (Proof.Events197.exact50665RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50664.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50664.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority50664.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50664.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority50664.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound50738

namespace LeftBound50739
def owner : Owner := ⟨.program ⟨214⟩, ⟨25764⟩⟩
def transferEvent : Nat := 50739
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50734 .summary) (.transfer 50738) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50734 .summary)
      LeftBound50733.bound (LeftBound50733.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13365⟩⟩) (rawTerms := some (Proof.Events198.exact50734RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50733.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50738)
      LeftBound50738.bound (LeftBound50738.actual selector witness) := by
  exact .transfer (LeftBound50738.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound50733.bound LeftBound50738.bound
def bound : CoeffClass := .finite ⟨350377660645376, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50733.bound, LeftBound50738.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound50733.actual selector witness) * (LeftBound50738.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50739

namespace LeftBound50750
def owner : Owner := ⟨.program ⟨214⟩, ⟨20254⟩⟩
def transferEvent : Nat := 50750
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 50748 .coefficient) (.value (.predecessor 1 50749 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50748 .coefficient)
      LeftAuthority50746.bound (LeftAuthority50746.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50747RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50746.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50746.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50749 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority50746.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50746.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority50746.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound50750

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
