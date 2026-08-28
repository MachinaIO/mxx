import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard183

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound27750
def owner : Owner := ⟨.program ⟨214⟩, ⟨13590⟩⟩
def transferEvent : Nat := 27750
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 27745 .summary) (.transfer 27749) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27745 .summary)
      LeftBound27743.bound (LeftBound27743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13589⟩⟩) (rawTerms := some (Proof.Events108.exact27745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 27749)
      LeftBound27749.bound (LeftBound27749.actual selector witness) := by
  exact .transfer (LeftBound27749.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound27743.bound LeftBound27749.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27743.bound, LeftBound27749.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound27743.actual selector witness) * (LeftBound27749.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27750

namespace LeftBound27758
def owner : Owner := ⟨.program ⟨214⟩, ⟨13591⟩⟩
def transferEvent : Nat := 27758
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 27756 .coefficient, .predecessor 1 27757 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27756 .coefficient)
      LeftBound27748.bound (LeftBound27748.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27755RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27748.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27748.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27757 .coefficient)
      LeftBound27720.bound (LeftBound27720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27720.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27720.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27748.bound, LeftBound27720.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27748.bound, LeftBound27720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27748.actual selector witness, LeftBound27720.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27758

namespace LeftBound27760
def owner : Owner := ⟨.program ⟨214⟩, ⟨13591⟩⟩
def transferEvent : Nat := 27760
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 27755 .summary, .result 27725 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27755 .summary)
      LeftBound27750.bound (LeftBound27750.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13590⟩⟩) (rawTerms := some (Proof.Events108.exact27755RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27750.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27725 .summary)
      LeftBound27722.bound (LeftBound27722.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13586⟩⟩) (rawTerms := some (Proof.Events108.exact27725RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27722.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27750.bound, LeftBound27722.bound]
def bound : CoeffClass := .finite ⟨95428736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27750.bound, LeftBound27722.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27750.actual selector witness, LeftBound27722.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27760

namespace LeftBound27764
def owner : Owner := ⟨.program ⟨214⟩, ⟨25851⟩⟩
def transferEvent : Nat := 27764
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 27762 .coefficient) (.predecessor 1 27763 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27762 .coefficient)
      LeftBound27758.bound (LeftBound27758.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27761RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27758.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27758.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27763 .coefficient)
      LeftAuthority27696.bound (LeftAuthority27696.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27697RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27696.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27696.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound27758.bound LeftAuthority27696.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27758.bound, LeftAuthority27696.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound27758.actual selector witness) * (LeftAuthority27696.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27764

namespace LeftBound27765
def owner : Owner := ⟨.program ⟨214⟩, ⟨25851⟩⟩
def transferEvent : Nat := 27765
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25850⟩⟩]⟩ [⟨.result 27697 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27697 .coefficient)
      LeftAuthority27696.bound (LeftAuthority27696.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25850⟩⟩) (rawTerms := some (Proof.Events108.exact27697RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27696.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27696.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority27696.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27696.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority27696.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound27765

namespace LeftBound27766
def owner : Owner := ⟨.program ⟨214⟩, ⟨25851⟩⟩
def transferEvent : Nat := 27766
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 27761 .summary) (.transfer 27765) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27761 .summary)
      LeftBound27760.bound (LeftBound27760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13591⟩⟩) (rawTerms := some (Proof.Events108.exact27761RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 27765)
      LeftBound27765.bound (LeftBound27765.actual selector witness) := by
  exact .transfer (LeftBound27765.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound27760.bound LeftBound27765.bound
def bound : CoeffClass := .finite ⟨350224987979776, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27760.bound, LeftBound27765.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound27760.actual selector witness) * (LeftBound27765.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27766

namespace LeftBound27777
def owner : Owner := ⟨.program ⟨214⟩, ⟨19326⟩⟩
def transferEvent : Nat := 27777
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 27775 .coefficient) (.value (.predecessor 1 27776 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27775 .coefficient)
      LeftAuthority27773.bound (LeftAuthority27773.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27774RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27773.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27773.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27776 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority27773.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27773.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority27773.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound27777

namespace LeftBound27781
def owner : Owner := ⟨.program ⟨214⟩, ⟨19327⟩⟩
def transferEvent : Nat := 27781
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 27779 .coefficient) (.predecessor 1 27780 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27779 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27780 .coefficient)
      LeftBound27777.bound (LeftBound27777.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27777.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27777.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound27777.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound27777.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound27777.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27781

namespace LeftBound27782
def owner : Owner := ⟨.program ⟨214⟩, ⟨19327⟩⟩
def transferEvent : Nat := 27782
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19324⟩⟩]⟩ [⟨.result 27774 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27774 .coefficient)
      LeftAuthority27773.bound (LeftAuthority27773.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19324⟩⟩) (rawTerms := some (Proof.Events108.exact27774RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27773.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27773.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority27773.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27773.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority27773.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound27782

namespace LeftBound27783
def owner : Owner := ⟨.program ⟨214⟩, ⟨19327⟩⟩
def transferEvent : Nat := 27783
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 27782) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 27782)
      LeftBound27782.bound (LeftBound27782.actual selector witness) := by
  exact .transfer (LeftBound27782.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound27782.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound27782.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound27782.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27783

namespace LeftBound27862
def owner : Owner := ⟨.program ⟨214⟩, ⟨13584⟩⟩
def transferEvent : Nat := 27862
def frameStart : Nat := 27833
def rule : BoundRule := .product (.predecessor 0 27860 .coefficient) (.predecessor 1 27861 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27860 .coefficient)
      LeftAuthority27858.bound (LeftAuthority27858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27858.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27858.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27861 .coefficient)
      LeftAuthority27855.bound (LeftAuthority27855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27855.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27855.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority27858.bound LeftAuthority27855.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27858.bound, LeftAuthority27855.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority27858.actual selector witness) * (LeftAuthority27855.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27862

namespace LeftBound27866
def owner : Owner := ⟨.program ⟨214⟩, ⟨13585⟩⟩
def transferEvent : Nat := 27866
def frameStart : Nat := 27833
def rule : BoundRule := .identity (.predecessor 0 27865 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27865 .coefficient)
      LeftBound27862.bound (LeftBound27862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27864RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27862.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27862.derived selector witness)

def rawBound : CoeffClass := LeftBound27862.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound27862.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound27866

namespace LeftBound27883
def owner : Owner := ⟨.program ⟨214⟩, ⟨13675⟩⟩
def transferEvent : Nat := 27883
def frameStart : Nat := 27833
def rule : BoundRule := .sum [.predecessor 0 27881 .coefficient, .predecessor 1 27882 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27881 .coefficient)
      LeftBound27866.bound (LeftBound27866.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound27866.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27882 .coefficient)
      LeftAuthority27879.bound (LeftAuthority27879.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority27879.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27866.bound, LeftAuthority27879.bound]
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27866.bound, LeftAuthority27879.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27866.actual selector witness, LeftAuthority27879.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27883

namespace LeftBound27886
def owner : Owner := ⟨.program ⟨214⟩, ⟨13676⟩⟩
def transferEvent : Nat := 27886
def frameStart : Nat := 27833
def rule : BoundRule := .identity (.predecessor 0 27885 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27885 .coefficient)
      LeftBound27883.bound (LeftBound27883.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound27883.derived selector witness)

def rawBound : CoeffClass := LeftBound27883.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27883.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound27883.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound27886

namespace LeftBound27892
def owner : Owner := ⟨.program ⟨214⟩, ⟨13677⟩⟩
def transferEvent : Nat := 27892
def frameStart : Nat := 27833
def rule : BoundRule := .product (.predecessor 0 27890 .coefficient) (.predecessor 1 27891 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27890 .coefficient)
      LeftAuthority27888.bound (LeftAuthority27888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27889RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27888.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27888.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27891 .coefficient)
      LeftBound27886.bound (LeftBound27886.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27887RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27886.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27886.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority27888.bound LeftBound27886.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27888.bound, LeftBound27886.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority27888.actual selector witness) * (LeftBound27886.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27892

namespace LeftBound27908
def owner : Owner := ⟨.program ⟨214⟩, ⟨7844⟩⟩
def transferEvent : Nat := 27908
def frameStart : Nat := 27833
def rule : BoundRule := .scale (.predecessor 0 27906 .coefficient) (.value (.predecessor 1 27907 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27906 .coefficient)
      LeftAuthority27904.bound (LeftAuthority27904.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact27905RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27904.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27904.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27907 .coefficient)
      LeftAuthority27895.bound (LeftAuthority27895.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority27895.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority27904.bound LeftAuthority27895.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27904.bound, LeftAuthority27895.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority27904.actual selector witness) * (LeftAuthority27895.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound27908

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
