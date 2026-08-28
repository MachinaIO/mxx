import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard072
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard174

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound26539
def owner : Owner := ⟨.program ⟨214⟩, ⟨21415⟩⟩
def transferEvent : Nat := 26539
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21412⟩⟩]⟩ [⟨.result 26531 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26531 .coefficient)
      LeftAuthority26530.bound (LeftAuthority26530.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21412⟩⟩) (rawTerms := some (Proof.Events103.exact26531RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26530.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26530.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority26530.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26530.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority26530.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound26539

namespace LeftBound26540
def owner : Owner := ⟨.program ⟨214⟩, ⟨21415⟩⟩
def transferEvent : Nat := 26540
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 26539) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 26539)
      LeftBound26539.bound (LeftBound26539.actual selector witness) := by
  exact .transfer (LeftBound26539.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound26539.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound26539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound26539.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26540

namespace LeftBound26635
def owner : Owner := ⟨.program ⟨214⟩, ⟨15953⟩⟩
def transferEvent : Nat := 26635
def frameStart : Nat := 26596
def rule : BoundRule := .identity (.predecessor 0 26634 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26634 .coefficient)
      LeftAuthority26632.bound (LeftAuthority26632.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26633RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26632.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26632.derived selector witness)

def rawBound : CoeffClass := LeftAuthority26632.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26632.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority26632.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound26635

namespace LeftBound26652
def owner : Owner := ⟨.program ⟨214⟩, ⟨16027⟩⟩
def transferEvent : Nat := 26652
def frameStart : Nat := 26596
def rule : BoundRule := .sum [.predecessor 0 26650 .coefficient, .predecessor 1 26651 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26650 .coefficient)
      LeftBound26635.bound (LeftBound26635.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound26635.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26651 .coefficient)
      LeftAuthority26648.bound (LeftAuthority26648.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority26648.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26635.bound, LeftAuthority26648.bound]
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26635.bound, LeftAuthority26648.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26635.actual selector witness, LeftAuthority26648.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26652

namespace LeftBound26655
def owner : Owner := ⟨.program ⟨214⟩, ⟨16028⟩⟩
def transferEvent : Nat := 26655
def frameStart : Nat := 26596
def rule : BoundRule := .identity (.predecessor 0 26654 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26654 .coefficient)
      LeftBound26652.bound (LeftBound26652.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound26652.derived selector witness)

def rawBound : CoeffClass := LeftBound26652.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26652.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound26652.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound26655

namespace LeftBound26661
def owner : Owner := ⟨.program ⟨214⟩, ⟨16029⟩⟩
def transferEvent : Nat := 26661
def frameStart : Nat := 26596
def rule : BoundRule := .product (.predecessor 0 26659 .coefficient) (.predecessor 1 26660 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26659 .coefficient)
      LeftAuthority26657.bound (LeftAuthority26657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26658RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26657.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26657.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26660 .coefficient)
      LeftBound26655.bound (LeftBound26655.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26656RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26655.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26655.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority26657.bound LeftBound26655.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26657.bound, LeftBound26655.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority26657.actual selector witness) * (LeftBound26655.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26661

namespace LeftBound26669
def owner : Owner := ⟨.program ⟨214⟩, ⟨16030⟩⟩
def transferEvent : Nat := 26669
def frameStart : Nat := 26596
def rule : BoundRule := .sum [.predecessor 0 26667 .coefficient, .predecessor 1 26668 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26667 .coefficient)
      LeftAuthority26665.bound (LeftAuthority26665.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26666RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26665.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26665.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26668 .coefficient)
      LeftBound26661.bound (LeftBound26661.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26661.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26661.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority26665.bound, LeftBound26661.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26665.bound, LeftBound26661.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority26665.actual selector witness, LeftBound26661.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26669

namespace LeftBound26673
def owner : Owner := ⟨.program ⟨214⟩, ⟨27906⟩⟩
def transferEvent : Nat := 26673
def frameStart : Nat := 26596
def rule : BoundRule := .product (.predecessor 0 26671 .coefficient) (.predecessor 1 26672 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26671 .coefficient)
      LeftBound26669.bound (LeftBound26669.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26669.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26669.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26672 .coefficient)
      LeftAuthority26646.bound (LeftAuthority26646.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26647RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26646.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26646.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound26669.bound LeftAuthority26646.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26669.bound, LeftAuthority26646.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound26669.actual selector witness) * (LeftAuthority26646.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26673

namespace LeftBound26684
def owner : Owner := ⟨.program ⟨214⟩, ⟨15996⟩⟩
def transferEvent : Nat := 26684
def frameStart : Nat := 26596
def rule : BoundRule := .product (.predecessor 0 26682 .coefficient) (.predecessor 1 26683 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26682 .coefficient)
      LeftAuthority26657.bound (LeftAuthority26657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26658RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26657.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26657.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26683 .coefficient)
      LeftAuthority26680.bound (LeftAuthority26680.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26681RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26680.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26680.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority26657.bound LeftAuthority26680.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26657.bound, LeftAuthority26680.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority26657.actual selector witness) * (LeftAuthority26680.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26684

namespace LeftBound26692
def owner : Owner := ⟨.program ⟨214⟩, ⟨15997⟩⟩
def transferEvent : Nat := 26692
def frameStart : Nat := 26596
def rule : BoundRule := .sum [.predecessor 0 26690 .coefficient, .predecessor 1 26691 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26690 .coefficient)
      LeftAuthority26688.bound (LeftAuthority26688.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26688.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26688.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26691 .coefficient)
      LeftBound26684.bound (LeftBound26684.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26686RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26684.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26684.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority26688.bound, LeftBound26684.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26688.bound, LeftBound26684.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority26688.actual selector witness, LeftBound26684.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26692

namespace LeftBound26696
def owner : Owner := ⟨.program ⟨214⟩, ⟨27910⟩⟩
def transferEvent : Nat := 26696
def frameStart : Nat := 26596
def rule : BoundRule := .sum [.predecessor 0 26694 .coefficient, .predecessor 1 26695 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26694 .coefficient)
      LeftBound26692.bound (LeftBound26692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26692.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26695 .coefficient)
      LeftBound26673.bound (LeftBound26673.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26678RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26673.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26673.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26692.bound, LeftBound26673.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26692.bound, LeftBound26673.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26692.actual selector witness, LeftBound26673.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26696

namespace LeftBound26709
def owner : Owner := ⟨.program ⟨214⟩, ⟨27908⟩⟩
def transferEvent : Nat := 26709
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26707 .coefficient, .predecessor 1 26708 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26707 .coefficient)
      LeftBound26538.bound (LeftBound26538.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26706RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26538.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26538.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26708 .coefficient)
      LeftBound26521.bound (LeftBound26521.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26521.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26521.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26538.bound, LeftBound26521.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26538.bound, LeftBound26521.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26538.actual selector witness, LeftBound26521.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26709

namespace LeftBound26712
def owner : Owner := ⟨.program ⟨214⟩, ⟨27908⟩⟩
def transferEvent : Nat := 26712
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26706 .summary, .result 26528 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26706 .summary)
      LeftBound26540.bound (LeftBound26540.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21415⟩⟩) (rawTerms := some (Proof.Events104.exact26706RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26540.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26528 .summary)
      LeftBound26523.bound (LeftBound26523.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27907⟩⟩) (rawTerms := some (Proof.Events103.exact26528RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26523.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26540.bound, LeftBound26523.bound]
def bound : CoeffClass := .finite ⟨1292068473939586330624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26540.bound, LeftBound26523.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26540.actual selector witness, LeftBound26523.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26712

namespace LeftBound26736
def owner : Owner := ⟨.program ⟨214⟩, ⟨11398⟩⟩
def transferEvent : Nat := 26736
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 26734 .coefficient) (.predecessor 1 26735 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26734 .coefficient)
      LeftAuthority1094.bound (LeftAuthority1094.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1095RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1094.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1094.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26735 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1094.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1094.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1094.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound26736

namespace LeftBound26741
def owner : Owner := ⟨.program ⟨214⟩, ⟨7348⟩⟩
def transferEvent : Nat := 26741
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 26739 .coefficient) (.predecessor 1 26740 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26739 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26740 .coefficient)
      LeftBound11982.bound (LeftBound11982.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11982.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11982.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound11982.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound11982.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound11982.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26741

namespace LeftBound26746
def owner : Owner := ⟨.program ⟨214⟩, ⟨11399⟩⟩
def transferEvent : Nat := 26746
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26744 .coefficient, .predecessor 1 26745 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26744 .coefficient)
      LeftBound26741.bound (LeftBound26741.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26743RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26741.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26741.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26745 .coefficient)
      LeftBound26736.bound (LeftBound26736.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26738RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26736.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26736.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26741.bound, LeftBound26736.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26741.bound, LeftBound26736.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26741.actual selector witness, LeftBound26736.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26746

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
