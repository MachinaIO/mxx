import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard053
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard054

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound9732
def owner : Owner := ⟨.program ⟨214⟩, ⟨16400⟩⟩
def transferEvent : Nat := 9732
def frameStart : Nat := 9622
def rule : BoundRule := .sum [.predecessor 0 9730 .coefficient, .predecessor 1 9731 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9730 .coefficient)
      LeftAuthority9728.bound (LeftAuthority9728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9729RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9728.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9731 .coefficient)
      LeftBound9724.bound (LeftBound9724.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9726RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9724.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9724.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority9728.bound, LeftBound9724.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9728.bound, LeftBound9724.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority9728.actual selector witness, LeftBound9724.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9732

namespace LeftBound9736
def owner : Owner := ⟨.program ⟨214⟩, ⟨25243⟩⟩
def transferEvent : Nat := 9736
def frameStart : Nat := 9622
def rule : BoundRule := .sum [.predecessor 0 9734 .coefficient, .predecessor 1 9735 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9734 .coefficient)
      LeftBound9732.bound (LeftBound9732.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9733RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9732.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9732.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9735 .coefficient)
      LeftBound9713.bound (LeftBound9713.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9718RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9713.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9713.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9732.bound, LeftBound9713.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9732.bound, LeftBound9713.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9732.actual selector witness, LeftBound9713.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9736

namespace LeftBound9749
def owner : Owner := ⟨.program ⟨214⟩, ⟨25241⟩⟩
def transferEvent : Nat := 9749
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9747 .coefficient, .predecessor 1 9748 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9747 .coefficient)
      LeftBound9570.bound (LeftBound9570.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9746RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9570.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9570.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9748 .coefficient)
      LeftBound9553.bound (LeftBound9553.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9560RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9553.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9553.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9570.bound, LeftBound9553.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9570.bound, LeftBound9553.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9570.actual selector witness, LeftBound9553.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9749

namespace LeftBound9752
def owner : Owner := ⟨.program ⟨214⟩, ⟨25241⟩⟩
def transferEvent : Nat := 9752
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 9746 .summary, .result 9560 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9746 .summary)
      LeftBound9572.bound (LeftBound9572.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19835⟩⟩) (rawTerms := some (Proof.Events038.exact9746RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound9572.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9560 .summary)
      LeftBound9555.bound (LeftBound9555.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25240⟩⟩) (rawTerms := some (Proof.Events037.exact9560RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound9555.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9572.bound, LeftBound9555.bound]
def bound : CoeffClass := .finite ⟨352115681275904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9572.bound, LeftBound9555.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9572.actual selector witness, LeftBound9555.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9752

namespace LeftBound9756
def owner : Owner := ⟨.program ⟨214⟩, ⟨28788⟩⟩
def transferEvent : Nat := 9756
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9754 .coefficient) (.predecessor 1 9755 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9754 .coefficient)
      LeftBound9749.bound (LeftBound9749.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9753RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9749.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9749.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9755 .coefficient)
      LeftAuthority9456.bound (LeftAuthority9456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9456.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9456.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound9749.bound LeftAuthority9456.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9749.bound, LeftAuthority9456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound9749.actual selector witness) * (LeftAuthority9456.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9756

namespace LeftBound9757
def owner : Owner := ⟨.program ⟨214⟩, ⟨28788⟩⟩
def transferEvent : Nat := 9757
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28786⟩⟩]⟩ [⟨.result 9457 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9457 .coefficient)
      LeftAuthority9456.bound (LeftAuthority9456.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28786⟩⟩) (rawTerms := some (Proof.Events036.exact9457RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9456.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9456.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority9456.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority9456.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound9757

namespace LeftBound9758
def owner : Owner := ⟨.program ⟨214⟩, ⟨28788⟩⟩
def transferEvent : Nat := 9758
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 9753 .summary) (.transfer 9757) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9753 .summary)
      LeftBound9752.bound (LeftBound9752.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25241⟩⟩) (rawTerms := some (Proof.Events038.exact9753RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound9752.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 9757)
      LeftBound9757.bound (LeftBound9757.actual selector witness) := by
  exact .transfer (LeftBound9757.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound9752.bound LeftBound9757.bound
def bound : CoeffClass := .finite ⟨1292270184133468094464, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9752.bound, LeftBound9757.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound9752.actual selector witness) * (LeftBound9757.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9758

namespace LeftBound9769
def owner : Owner := ⟨.program ⟨214⟩, ⟨21994⟩⟩
def transferEvent : Nat := 9769
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 9767 .coefficient) (.value (.predecessor 1 9768 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9767 .coefficient)
      LeftAuthority9765.bound (LeftAuthority9765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9766RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9765.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9765.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9768 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority9765.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9765.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority9765.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound9769

namespace LeftBound9773
def owner : Owner := ⟨.program ⟨214⟩, ⟨21995⟩⟩
def transferEvent : Nat := 9773
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9771 .coefficient) (.predecessor 1 9772 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9771 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9772 .coefficient)
      LeftBound9769.bound (LeftBound9769.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9770RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9769.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9769.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound9769.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound9769.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound9769.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9773

namespace LeftBound9774
def owner : Owner := ⟨.program ⟨214⟩, ⟨21995⟩⟩
def transferEvent : Nat := 9774
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21992⟩⟩]⟩ [⟨.result 9766 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9766 .coefficient)
      LeftAuthority9765.bound (LeftAuthority9765.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21992⟩⟩) (rawTerms := some (Proof.Events038.exact9766RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9765.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9765.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority9765.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9765.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority9765.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound9774

namespace LeftBound9775
def owner : Owner := ⟨.program ⟨214⟩, ⟨21995⟩⟩
def transferEvent : Nat := 9775
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 9774) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 9774)
      LeftBound9774.bound (LeftBound9774.actual selector witness) := by
  exact .transfer (LeftBound9774.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound9774.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound9774.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound9774.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9775

namespace LeftBound9870
def owner : Owner := ⟨.program ⟨214⟩, ⟨16398⟩⟩
def transferEvent : Nat := 9870
def frameStart : Nat := 9831
def rule : BoundRule := .identity (.predecessor 0 9869 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9869 .coefficient)
      LeftAuthority9867.bound (LeftAuthority9867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9867.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9867.derived selector witness)

def rawBound : CoeffClass := LeftAuthority9867.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9867.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority9867.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound9870

namespace LeftBound9887
def owner : Owner := ⟨.program ⟨214⟩, ⟨16437⟩⟩
def transferEvent : Nat := 9887
def frameStart : Nat := 9831
def rule : BoundRule := .sum [.predecessor 0 9885 .coefficient, .predecessor 1 9886 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9885 .coefficient)
      LeftBound9870.bound (LeftBound9870.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound9870.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9886 .coefficient)
      LeftAuthority9883.bound (LeftAuthority9883.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority9883.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9870.bound, LeftAuthority9883.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9870.bound, LeftAuthority9883.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9870.actual selector witness, LeftAuthority9883.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9887

namespace LeftBound9890
def owner : Owner := ⟨.program ⟨214⟩, ⟨16438⟩⟩
def transferEvent : Nat := 9890
def frameStart : Nat := 9831
def rule : BoundRule := .identity (.predecessor 0 9889 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9889 .coefficient)
      LeftBound9887.bound (LeftBound9887.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound9887.derived selector witness)

def rawBound : CoeffClass := LeftBound9887.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9887.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound9887.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound9890

namespace LeftBound9896
def owner : Owner := ⟨.program ⟨214⟩, ⟨16439⟩⟩
def transferEvent : Nat := 9896
def frameStart : Nat := 9831
def rule : BoundRule := .product (.predecessor 0 9894 .coefficient) (.predecessor 1 9895 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9894 .coefficient)
      LeftAuthority9892.bound (LeftAuthority9892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9892.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9892.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9895 .coefficient)
      LeftBound9890.bound (LeftBound9890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9890.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9890.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority9892.bound LeftBound9890.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9892.bound, LeftBound9890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority9892.actual selector witness) * (LeftBound9890.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9896

namespace LeftBound9904
def owner : Owner := ⟨.program ⟨214⟩, ⟨16440⟩⟩
def transferEvent : Nat := 9904
def frameStart : Nat := 9831
def rule : BoundRule := .sum [.predecessor 0 9902 .coefficient, .predecessor 1 9903 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9902 .coefficient)
      LeftAuthority9900.bound (LeftAuthority9900.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9901RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9900.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9900.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9903 .coefficient)
      LeftBound9896.bound (LeftBound9896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9896.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9896.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority9900.bound, LeftBound9896.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9900.bound, LeftBound9896.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority9900.actual selector witness, LeftBound9896.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9904

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
