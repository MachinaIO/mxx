import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard053

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound9555
def owner : Owner := ⟨.program ⟨214⟩, ⟨25240⟩⟩
def transferEvent : Nat := 9555
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 9550 .summary) (.transfer 9554) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9550 .summary)
      LeftBound9549.bound (LeftBound9549.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11996⟩⟩) (rawTerms := some (Proof.Events037.exact9550RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound9549.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 9554)
      LeftBound9554.bound (LeftBound9554.actual selector witness) := by
  exact .transfer (LeftBound9554.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound9549.bound LeftBound9554.bound
def bound : CoeffClass := .finite ⟨350304377765888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9549.bound, LeftBound9554.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound9549.actual selector witness) * (LeftBound9554.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9555

namespace LeftBound9566
def owner : Owner := ⟨.program ⟨214⟩, ⟨19834⟩⟩
def transferEvent : Nat := 9566
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 9564 .coefficient) (.value (.predecessor 1 9565 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9564 .coefficient)
      LeftAuthority9562.bound (LeftAuthority9562.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9562.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9562.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9565 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority9562.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9562.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority9562.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound9566

namespace LeftBound9570
def owner : Owner := ⟨.program ⟨214⟩, ⟨19835⟩⟩
def transferEvent : Nat := 9570
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9568 .coefficient) (.predecessor 1 9569 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9568 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9569 .coefficient)
      LeftBound9566.bound (LeftBound9566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9567RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9566.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9566.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound9566.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound9566.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound9566.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9570

namespace LeftBound9571
def owner : Owner := ⟨.program ⟨214⟩, ⟨19835⟩⟩
def transferEvent : Nat := 9571
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19832⟩⟩]⟩ [⟨.result 9563 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9563 .coefficient)
      LeftAuthority9562.bound (LeftAuthority9562.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19832⟩⟩) (rawTerms := some (Proof.Events037.exact9563RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9562.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9562.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority9562.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority9562.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound9571

namespace LeftBound9572
def owner : Owner := ⟨.program ⟨214⟩, ⟨19835⟩⟩
def transferEvent : Nat := 9572
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 9571) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 9571)
      LeftBound9571.bound (LeftBound9571.actual selector witness) := by
  exact .transfer (LeftBound9571.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound9571.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound9571.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound9571.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9572

namespace LeftBound9651
def owner : Owner := ⟨.program ⟨214⟩, ⟨11990⟩⟩
def transferEvent : Nat := 9651
def frameStart : Nat := 9622
def rule : BoundRule := .product (.predecessor 0 9649 .coefficient) (.predecessor 1 9650 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9649 .coefficient)
      LeftAuthority9647.bound (LeftAuthority9647.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9648RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9647.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9647.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9650 .coefficient)
      LeftAuthority9644.bound (LeftAuthority9644.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9645RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9644.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9644.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority9647.bound LeftAuthority9644.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9647.bound, LeftAuthority9644.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority9647.actual selector witness) * (LeftAuthority9644.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9651

namespace LeftBound9655
def owner : Owner := ⟨.program ⟨214⟩, ⟨11991⟩⟩
def transferEvent : Nat := 9655
def frameStart : Nat := 9622
def rule : BoundRule := .identity (.predecessor 0 9654 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9654 .coefficient)
      LeftBound9651.bound (LeftBound9651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9651.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9651.derived selector witness)

def rawBound : CoeffClass := LeftBound9651.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9651.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound9651.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound9655

namespace LeftBound9672
def owner : Owner := ⟨.program ⟨214⟩, ⟨12069⟩⟩
def transferEvent : Nat := 9672
def frameStart : Nat := 9622
def rule : BoundRule := .sum [.predecessor 0 9670 .coefficient, .predecessor 1 9671 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9670 .coefficient)
      LeftBound9655.bound (LeftBound9655.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound9655.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9671 .coefficient)
      LeftAuthority9668.bound (LeftAuthority9668.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority9668.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9655.bound, LeftAuthority9668.bound]
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9655.bound, LeftAuthority9668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9655.actual selector witness, LeftAuthority9668.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9672

namespace LeftBound9675
def owner : Owner := ⟨.program ⟨214⟩, ⟨12070⟩⟩
def transferEvent : Nat := 9675
def frameStart : Nat := 9622
def rule : BoundRule := .identity (.predecessor 0 9674 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9674 .coefficient)
      LeftBound9672.bound (LeftBound9672.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound9672.derived selector witness)

def rawBound : CoeffClass := LeftBound9672.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9672.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound9672.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound9675

namespace LeftBound9681
def owner : Owner := ⟨.program ⟨214⟩, ⟨12071⟩⟩
def transferEvent : Nat := 9681
def frameStart : Nat := 9622
def rule : BoundRule := .product (.predecessor 0 9679 .coefficient) (.predecessor 1 9680 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9679 .coefficient)
      LeftAuthority9677.bound (LeftAuthority9677.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9678RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9677.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9677.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9680 .coefficient)
      LeftBound9675.bound (LeftBound9675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9676RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9675.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9675.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority9677.bound LeftBound9675.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9677.bound, LeftBound9675.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority9677.actual selector witness) * (LeftBound9675.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9681

namespace LeftBound9697
def owner : Owner := ⟨.program ⟨214⟩, ⟨7865⟩⟩
def transferEvent : Nat := 9697
def frameStart : Nat := 9622
def rule : BoundRule := .scale (.predecessor 0 9695 .coefficient) (.value (.predecessor 1 9696 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9695 .coefficient)
      LeftAuthority9693.bound (LeftAuthority9693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9694RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9693.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9693.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9696 .coefficient)
      LeftAuthority9684.bound (LeftAuthority9684.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority9684.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority9693.bound LeftAuthority9684.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9693.bound, LeftAuthority9684.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority9693.actual selector witness) * (LeftAuthority9684.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound9697

namespace LeftBound9700
def owner : Owner := ⟨.program ⟨214⟩, ⟨6764⟩⟩
def transferEvent : Nat := 9700
def frameStart : Nat := 9622
def rule : BoundRule := .identity (.predecessor 0 9699 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9699 .coefficient)
      LeftAuthority9687.bound (LeftAuthority9687.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9688RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9687.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9687.derived selector witness)

def rawBound : CoeffClass := LeftAuthority9687.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9687.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority9687.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound9700

namespace LeftBound9704
def owner : Owner := ⟨.program ⟨214⟩, ⟨7866⟩⟩
def transferEvent : Nat := 9704
def frameStart : Nat := 9622
def rule : BoundRule := .product (.predecessor 0 9702 .coefficient) (.predecessor 1 9703 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9702 .coefficient)
      LeftBound9700.bound (LeftBound9700.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9701RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9700.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9700.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9703 .coefficient)
      LeftBound9697.bound (LeftBound9697.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9698RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9697.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9697.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound9700.bound LeftBound9697.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9700.bound, LeftBound9697.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound9700.actual selector witness) * (LeftBound9697.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9704

namespace LeftBound9709
def owner : Owner := ⟨.program ⟨214⟩, ⟨12072⟩⟩
def transferEvent : Nat := 9709
def frameStart : Nat := 9622
def rule : BoundRule := .sum [.predecessor 0 9707 .coefficient, .predecessor 1 9708 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9707 .coefficient)
      LeftBound9704.bound (LeftBound9704.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9706RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9704.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9704.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9708 .coefficient)
      LeftBound9681.bound (LeftBound9681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9681.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9681.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9704.bound, LeftBound9681.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9704.bound, LeftBound9681.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9704.actual selector witness, LeftBound9681.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9709

namespace LeftBound9713
def owner : Owner := ⟨.program ⟨214⟩, ⟨25242⟩⟩
def transferEvent : Nat := 9713
def frameStart : Nat := 9622
def rule : BoundRule := .product (.predecessor 0 9711 .coefficient) (.predecessor 1 9712 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9711 .coefficient)
      LeftBound9709.bound (LeftBound9709.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9710RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9709.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9709.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9712 .coefficient)
      LeftAuthority9666.bound (LeftAuthority9666.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9667RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9666.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9666.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound9709.bound LeftAuthority9666.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9709.bound, LeftAuthority9666.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound9709.actual selector witness) * (LeftAuthority9666.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9713

namespace LeftBound9724
def owner : Owner := ⟨.program ⟨214⟩, ⟨16399⟩⟩
def transferEvent : Nat := 9724
def frameStart : Nat := 9622
def rule : BoundRule := .product (.predecessor 0 9722 .coefficient) (.predecessor 1 9723 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9722 .coefficient)
      LeftAuthority9677.bound (LeftAuthority9677.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9678RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9677.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9677.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9723 .coefficient)
      LeftAuthority9720.bound (LeftAuthority9720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9721RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9720.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9720.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority9677.bound LeftAuthority9720.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9677.bound, LeftAuthority9720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority9677.actual selector witness) * (LeftAuthority9720.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9724

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
