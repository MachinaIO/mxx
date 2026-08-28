import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard152

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound23630
def owner : Owner := ⟨.program ⟨214⟩, ⟨29209⟩⟩
def transferEvent : Nat := 23630
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29207⟩⟩]⟩ [⟨.result 23349 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23349 .coefficient)
      LeftAuthority23348.bound (LeftAuthority23348.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29207⟩⟩) (rawTerms := some (Proof.Events091.exact23349RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23348.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23348.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority23348.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23348.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority23348.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound23630

namespace LeftBound23631
def owner : Owner := ⟨.program ⟨214⟩, ⟨29209⟩⟩
def transferEvent : Nat := 23631
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 23626 .summary) (.transfer 23630) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23626 .summary)
      LeftBound23625.bound (LeftBound23625.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25467⟩⟩) (rawTerms := some (Proof.Events092.exact23626RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23625.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 23630)
      LeftBound23630.bound (LeftBound23630.actual selector witness) := by
  exact .transfer (LeftBound23630.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound23625.bound LeftBound23630.bound
def bound : CoeffClass := .finite ⟨1292337421468529852416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23625.bound, LeftBound23630.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound23625.actual selector witness) * (LeftBound23630.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23631

namespace LeftBound23642
def owner : Owner := ⟨.program ⟨214⟩, ⟨22278⟩⟩
def transferEvent : Nat := 23642
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 23640 .coefficient) (.value (.predecessor 1 23641 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23640 .coefficient)
      LeftAuthority23638.bound (LeftAuthority23638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23639RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23638.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23638.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23641 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority23638.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23638.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority23638.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound23642

namespace LeftBound23646
def owner : Owner := ⟨.program ⟨214⟩, ⟨22279⟩⟩
def transferEvent : Nat := 23646
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 23644 .coefficient) (.predecessor 1 23645 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23644 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23645 .coefficient)
      LeftBound23642.bound (LeftBound23642.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23643RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23642.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23642.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound23642.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound23642.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound23642.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23646

namespace LeftBound23647
def owner : Owner := ⟨.program ⟨214⟩, ⟨22279⟩⟩
def transferEvent : Nat := 23647
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22276⟩⟩]⟩ [⟨.result 23639 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23639 .coefficient)
      LeftAuthority23638.bound (LeftAuthority23638.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22276⟩⟩) (rawTerms := some (Proof.Events092.exact23639RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23638.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23638.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority23638.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23638.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority23638.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound23647

namespace LeftBound23648
def owner : Owner := ⟨.program ⟨214⟩, ⟨22279⟩⟩
def transferEvent : Nat := 23648
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 23647) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 23647)
      LeftBound23647.bound (LeftBound23647.actual selector witness) := by
  exact .transfer (LeftBound23647.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound23647.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound23647.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound23647.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23648

namespace LeftBound23743
def owner : Owner := ⟨.program ⟨214⟩, ⟨16562⟩⟩
def transferEvent : Nat := 23743
def frameStart : Nat := 23704
def rule : BoundRule := .identity (.predecessor 0 23742 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23742 .coefficient)
      LeftAuthority23740.bound (LeftAuthority23740.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23741RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23740.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23740.derived selector witness)

def rawBound : CoeffClass := LeftAuthority23740.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23740.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority23740.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound23743

namespace LeftBound23760
def owner : Owner := ⟨.program ⟨214⟩, ⟨16601⟩⟩
def transferEvent : Nat := 23760
def frameStart : Nat := 23704
def rule : BoundRule := .sum [.predecessor 0 23758 .coefficient, .predecessor 1 23759 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23758 .coefficient)
      LeftBound23743.bound (LeftBound23743.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound23743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23759 .coefficient)
      LeftAuthority23756.bound (LeftAuthority23756.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority23756.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23743.bound, LeftAuthority23756.bound]
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23743.bound, LeftAuthority23756.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23743.actual selector witness, LeftAuthority23756.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23760

namespace LeftBound23763
def owner : Owner := ⟨.program ⟨214⟩, ⟨16602⟩⟩
def transferEvent : Nat := 23763
def frameStart : Nat := 23704
def rule : BoundRule := .identity (.predecessor 0 23762 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23762 .coefficient)
      LeftBound23760.bound (LeftBound23760.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound23760.derived selector witness)

def rawBound : CoeffClass := LeftBound23760.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23760.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound23760.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound23763

namespace LeftBound23769
def owner : Owner := ⟨.program ⟨214⟩, ⟨16603⟩⟩
def transferEvent : Nat := 23769
def frameStart : Nat := 23704
def rule : BoundRule := .product (.predecessor 0 23767 .coefficient) (.predecessor 1 23768 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23767 .coefficient)
      LeftAuthority23765.bound (LeftAuthority23765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23766RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23765.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23765.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23768 .coefficient)
      LeftBound23763.bound (LeftBound23763.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23764RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23763.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23763.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority23765.bound LeftBound23763.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23765.bound, LeftBound23763.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority23765.actual selector witness) * (LeftBound23763.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23769

namespace LeftBound23777
def owner : Owner := ⟨.program ⟨214⟩, ⟨16604⟩⟩
def transferEvent : Nat := 23777
def frameStart : Nat := 23704
def rule : BoundRule := .sum [.predecessor 0 23775 .coefficient, .predecessor 1 23776 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23775 .coefficient)
      LeftAuthority23773.bound (LeftAuthority23773.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23774RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23773.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23773.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23776 .coefficient)
      LeftBound23769.bound (LeftBound23769.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23771RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23769.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23769.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority23773.bound, LeftBound23769.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23773.bound, LeftBound23769.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority23773.actual selector witness, LeftBound23769.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23777

namespace LeftBound23781
def owner : Owner := ⟨.program ⟨214⟩, ⟨29208⟩⟩
def transferEvent : Nat := 23781
def frameStart : Nat := 23704
def rule : BoundRule := .product (.predecessor 0 23779 .coefficient) (.predecessor 1 23780 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23779 .coefficient)
      LeftBound23777.bound (LeftBound23777.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23777.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23777.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23780 .coefficient)
      LeftAuthority23754.bound (LeftAuthority23754.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23755RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23754.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23754.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound23777.bound LeftAuthority23754.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23777.bound, LeftAuthority23754.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound23777.actual selector witness) * (LeftAuthority23754.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23781

namespace LeftBound23792
def owner : Owner := ⟨.program ⟨214⟩, ⟨18215⟩⟩
def transferEvent : Nat := 23792
def frameStart : Nat := 23704
def rule : BoundRule := .product (.predecessor 0 23790 .coefficient) (.predecessor 1 23791 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23790 .coefficient)
      LeftAuthority23765.bound (LeftAuthority23765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23766RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23765.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23765.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23791 .coefficient)
      LeftAuthority23788.bound (LeftAuthority23788.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23789RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23788.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23788.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority23765.bound LeftAuthority23788.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23765.bound, LeftAuthority23788.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority23765.actual selector witness) * (LeftAuthority23788.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23792

namespace LeftBound23800
def owner : Owner := ⟨.program ⟨214⟩, ⟨18216⟩⟩
def transferEvent : Nat := 23800
def frameStart : Nat := 23704
def rule : BoundRule := .sum [.predecessor 0 23798 .coefficient, .predecessor 1 23799 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23798 .coefficient)
      LeftAuthority23796.bound (LeftAuthority23796.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23797RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23796.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23796.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23799 .coefficient)
      LeftBound23792.bound (LeftBound23792.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23794RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23792.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23792.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority23796.bound, LeftBound23792.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23796.bound, LeftBound23792.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority23796.actual selector witness, LeftBound23792.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23800

namespace LeftBound23804
def owner : Owner := ⟨.program ⟨214⟩, ⟨29212⟩⟩
def transferEvent : Nat := 23804
def frameStart : Nat := 23704
def rule : BoundRule := .sum [.predecessor 0 23802 .coefficient, .predecessor 1 23803 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23802 .coefficient)
      LeftBound23800.bound (LeftBound23800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23801RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23800.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23800.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23803 .coefficient)
      LeftBound23781.bound (LeftBound23781.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23786RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23781.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23781.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23800.bound, LeftBound23781.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23800.bound, LeftBound23781.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23800.actual selector witness, LeftBound23781.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23804

namespace LeftBound23817
def owner : Owner := ⟨.program ⟨214⟩, ⟨29210⟩⟩
def transferEvent : Nat := 23817
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 23815 .coefficient, .predecessor 1 23816 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23815 .coefficient)
      LeftBound23646.bound (LeftBound23646.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact23814RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23646.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23646.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23816 .coefficient)
      LeftBound23629.bound (LeftBound23629.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23629.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23629.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23646.bound, LeftBound23629.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23646.bound, LeftBound23629.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23646.actual selector witness, LeftBound23629.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23817

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
