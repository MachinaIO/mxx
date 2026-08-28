import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard370
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard419

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound62657
def owner : Owner := ⟨.program ⟨214⟩, ⟨28527⟩⟩
def transferEvent : Nat := 62657
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩ [⟨.result 5655 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5655 .coefficient)
      LeftAuthority5654.bound (LeftAuthority5654.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6677⟩⟩) (rawTerms := some (Proof.Events022.exact5655RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5654.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5654.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5654.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5654.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5654.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound62657

namespace LeftBound62658
def owner : Owner := ⟨.program ⟨214⟩, ⟨28527⟩⟩
def transferEvent : Nat := 62658
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 62653 .summary) (.transfer 62657) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62653 .summary)
      LeftBound62652.bound (LeftBound62652.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28526⟩⟩) (rawTerms := some (Proof.Events244.exact62653RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62652.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 62657)
      LeftBound62657.bound (LeftBound62657.actual selector witness) := by
  exact .transfer (LeftBound62657.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound62652.bound LeftBound62657.bound
def bound : CoeffClass := .finite ⟨4742405496644812892115304448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62652.bound, LeftBound62657.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound62652.actual selector witness) * (LeftBound62657.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62658

namespace LeftBound62673
def owner : Owner := ⟨.program ⟨214⟩, ⟨28308⟩⟩
def transferEvent : Nat := 62673
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 62671 .coefficient) (.predecessor 1 62672 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62671 .coefficient)
      LeftBound54800.bound (LeftBound54800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54804RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54800.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54800.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62672 .coefficient)
      LeftAuthority62669.bound (LeftAuthority62669.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62669.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62669.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound54800.bound LeftAuthority62669.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54800.bound, LeftAuthority62669.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound54800.actual selector witness) * (LeftAuthority62669.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62673

namespace LeftBound62674
def owner : Owner := ⟨.program ⟨214⟩, ⟨28308⟩⟩
def transferEvent : Nat := 62674
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28306⟩⟩]⟩ [⟨.result 62670 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62670 .coefficient)
      LeftAuthority62669.bound (LeftAuthority62669.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28306⟩⟩) (rawTerms := some (Proof.Events244.exact62670RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62669.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62669.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority62669.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62669.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority62669.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound62674

namespace LeftBound62675
def owner : Owner := ⟨.program ⟨214⟩, ⟨28308⟩⟩
def transferEvent : Nat := 62675
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 54804 .summary) (.transfer 62674) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54804 .summary)
      LeftBound54803.bound (LeftBound54803.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26227⟩⟩) (rawTerms := some (Proof.Events214.exact54804RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54803.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 62674)
      LeftBound62674.bound (LeftBound62674.actual selector witness) := by
  exact .transfer (LeftBound62674.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound54803.bound LeftBound62674.bound
def bound : CoeffClass := .finite ⟨1292180534353385750528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54803.bound, LeftBound62674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound54803.actual selector witness) * (LeftBound62674.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62675

namespace LeftBound62686
def owner : Owner := ⟨.program ⟨214⟩, ⟨21622⟩⟩
def transferEvent : Nat := 62686
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 62684 .coefficient) (.value (.predecessor 1 62685 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62684 .coefficient)
      LeftAuthority62682.bound (LeftAuthority62682.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62682.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62685 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority62682.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62682.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority62682.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound62686

namespace LeftBound62690
def owner : Owner := ⟨.program ⟨214⟩, ⟨21623⟩⟩
def transferEvent : Nat := 62690
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 62688 .coefficient) (.predecessor 1 62689 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62688 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62689 .coefficient)
      LeftBound62686.bound (LeftBound62686.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62687RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62686.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62686.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound62686.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound62686.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound62686.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62690

namespace LeftBound62691
def owner : Owner := ⟨.program ⟨214⟩, ⟨21623⟩⟩
def transferEvent : Nat := 62691
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21620⟩⟩]⟩ [⟨.result 62683 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62683 .coefficient)
      LeftAuthority62682.bound (LeftAuthority62682.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21620⟩⟩) (rawTerms := some (Proof.Events244.exact62683RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62682.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority62682.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62682.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority62682.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound62691

namespace LeftBound62692
def owner : Owner := ⟨.program ⟨214⟩, ⟨21623⟩⟩
def transferEvent : Nat := 62692
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 62691) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 62691)
      LeftBound62691.bound (LeftBound62691.actual selector witness) := by
  exact .transfer (LeftBound62691.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound62691.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound62691.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound62691.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62692

namespace LeftBound62787
def owner : Owner := ⟨.program ⟨214⟩, ⟨16183⟩⟩
def transferEvent : Nat := 62787
def frameStart : Nat := 62748
def rule : BoundRule := .identity (.predecessor 0 62786 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62786 .coefficient)
      LeftAuthority62784.bound (LeftAuthority62784.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events245.exact62785RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62784.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62784.derived selector witness)

def rawBound : CoeffClass := LeftAuthority62784.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62784.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority62784.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound62787

namespace LeftBound62804
def owner : Owner := ⟨.program ⟨214⟩, ⟨16222⟩⟩
def transferEvent : Nat := 62804
def frameStart : Nat := 62748
def rule : BoundRule := .sum [.predecessor 0 62802 .coefficient, .predecessor 1 62803 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62802 .coefficient)
      LeftBound62787.bound (LeftBound62787.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound62787.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62803 .coefficient)
      LeftAuthority62800.bound (LeftAuthority62800.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority62800.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62787.bound, LeftAuthority62800.bound]
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62787.bound, LeftAuthority62800.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound62787.actual selector witness, LeftAuthority62800.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62804

namespace LeftBound62807
def owner : Owner := ⟨.program ⟨214⟩, ⟨16223⟩⟩
def transferEvent : Nat := 62807
def frameStart : Nat := 62748
def rule : BoundRule := .identity (.predecessor 0 62806 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62806 .coefficient)
      LeftBound62804.bound (LeftBound62804.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound62804.derived selector witness)

def rawBound : CoeffClass := LeftBound62804.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62804.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound62804.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound62807

namespace LeftBound62813
def owner : Owner := ⟨.program ⟨214⟩, ⟨16224⟩⟩
def transferEvent : Nat := 62813
def frameStart : Nat := 62748
def rule : BoundRule := .product (.predecessor 0 62811 .coefficient) (.predecessor 1 62812 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62811 .coefficient)
      LeftAuthority62809.bound (LeftAuthority62809.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events245.exact62810RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62809.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62809.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62812 .coefficient)
      LeftBound62807.bound (LeftBound62807.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events245.exact62808RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62807.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62807.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority62809.bound LeftBound62807.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62809.bound, LeftBound62807.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority62809.actual selector witness) * (LeftBound62807.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62813

namespace LeftBound62821
def owner : Owner := ⟨.program ⟨214⟩, ⟨16225⟩⟩
def transferEvent : Nat := 62821
def frameStart : Nat := 62748
def rule : BoundRule := .sum [.predecessor 0 62819 .coefficient, .predecessor 1 62820 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62819 .coefficient)
      LeftAuthority62817.bound (LeftAuthority62817.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events245.exact62818RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62817.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62817.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62820 .coefficient)
      LeftBound62813.bound (LeftBound62813.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events245.exact62815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62813.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62813.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority62817.bound, LeftBound62813.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62817.bound, LeftBound62813.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority62817.actual selector witness, LeftBound62813.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62821

namespace LeftBound62825
def owner : Owner := ⟨.program ⟨214⟩, ⟨28307⟩⟩
def transferEvent : Nat := 62825
def frameStart : Nat := 62748
def rule : BoundRule := .product (.predecessor 0 62823 .coefficient) (.predecessor 1 62824 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62823 .coefficient)
      LeftBound62821.bound (LeftBound62821.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events245.exact62822RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62821.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62821.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62824 .coefficient)
      LeftAuthority62798.bound (LeftAuthority62798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events245.exact62799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62798.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62798.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound62821.bound LeftAuthority62798.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62821.bound, LeftAuthority62798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound62821.actual selector witness) * (LeftAuthority62798.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62825

namespace LeftBound62836
def owner : Owner := ⟨.program ⟨214⟩, ⟨17668⟩⟩
def transferEvent : Nat := 62836
def frameStart : Nat := 62748
def rule : BoundRule := .product (.predecessor 0 62834 .coefficient) (.predecessor 1 62835 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62834 .coefficient)
      LeftAuthority62809.bound (LeftAuthority62809.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events245.exact62810RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62809.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62809.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62835 .coefficient)
      LeftAuthority62832.bound (LeftAuthority62832.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events245.exact62833RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62832.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62832.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority62809.bound LeftAuthority62832.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62809.bound, LeftAuthority62832.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority62809.actual selector witness) * (LeftAuthority62832.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62836

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
