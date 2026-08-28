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

namespace LeftBound32575
def owner : Owner := ⟨.program ⟨214⟩, ⟨29202⟩⟩
def transferEvent : Nat := 32575
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 32573 .coefficient) (.predecessor 1 32574 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32573 .coefficient)
      LeftBound23622.bound (LeftBound23622.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23626RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23622.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23622.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32574 .coefficient)
      LeftAuthority32571.bound (LeftAuthority32571.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32572RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32571.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32571.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound23622.bound LeftAuthority32571.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23622.bound, LeftAuthority32571.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound23622.actual selector witness) * (LeftAuthority32571.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32575

namespace LeftBound32576
def owner : Owner := ⟨.program ⟨214⟩, ⟨29202⟩⟩
def transferEvent : Nat := 32576
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29200⟩⟩]⟩ [⟨.result 32572 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 32572 .coefficient)
      LeftAuthority32571.bound (LeftAuthority32571.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29200⟩⟩) (rawTerms := some (Proof.Events127.exact32572RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32571.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32571.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority32571.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32571.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority32571.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound32576

namespace LeftBound32577
def owner : Owner := ⟨.program ⟨214⟩, ⟨29202⟩⟩
def transferEvent : Nat := 32577
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 23626 .summary) (.transfer 32576) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23626 .summary)
      LeftBound23625.bound (LeftBound23625.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25467⟩⟩) (rawTerms := some (Proof.Events092.exact23626RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23625.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 32576)
      LeftBound32576.bound (LeftBound32576.actual selector witness) := by
  exact .transfer (LeftBound32576.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound23625.bound LeftBound32576.bound
def bound : CoeffClass := .finite ⟨1292337421468529852416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23625.bound, LeftBound32576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound23625.actual selector witness) * (LeftBound32576.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32577

namespace LeftBound32588
def owner : Owner := ⟨.program ⟨214⟩, ⟨22206⟩⟩
def transferEvent : Nat := 32588
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 32586 .coefficient) (.value (.predecessor 1 32587 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32586 .coefficient)
      LeftAuthority32584.bound (LeftAuthority32584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32585RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32584.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32584.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32587 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority32584.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32584.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority32584.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound32588

namespace LeftBound32592
def owner : Owner := ⟨.program ⟨214⟩, ⟨22207⟩⟩
def transferEvent : Nat := 32592
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 32590 .coefficient) (.predecessor 1 32591 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32590 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32591 .coefficient)
      LeftBound32588.bound (LeftBound32588.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32588.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32588.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound32588.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound32588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound32588.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32592

namespace LeftBound32593
def owner : Owner := ⟨.program ⟨214⟩, ⟨22207⟩⟩
def transferEvent : Nat := 32593
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22204⟩⟩]⟩ [⟨.result 32585 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 32585 .coefficient)
      LeftAuthority32584.bound (LeftAuthority32584.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22204⟩⟩) (rawTerms := some (Proof.Events127.exact32585RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32584.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32584.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority32584.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32584.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority32584.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound32593

namespace LeftBound32594
def owner : Owner := ⟨.program ⟨214⟩, ⟨22207⟩⟩
def transferEvent : Nat := 32594
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 32593) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 32593)
      LeftBound32593.bound (LeftBound32593.actual selector witness) := by
  exact .transfer (LeftBound32593.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound32593.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound32593.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound32593.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32594

namespace LeftBound32689
def owner : Owner := ⟨.program ⟨214⟩, ⟨16562⟩⟩
def transferEvent : Nat := 32689
def frameStart : Nat := 32650
def rule : BoundRule := .identity (.predecessor 0 32688 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32688 .coefficient)
      LeftAuthority32686.bound (LeftAuthority32686.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32687RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32686.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32686.derived selector witness)

def rawBound : CoeffClass := LeftAuthority32686.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32686.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority32686.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound32689

namespace LeftBound32706
def owner : Owner := ⟨.program ⟨214⟩, ⟨16601⟩⟩
def transferEvent : Nat := 32706
def frameStart : Nat := 32650
def rule : BoundRule := .sum [.predecessor 0 32704 .coefficient, .predecessor 1 32705 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32704 .coefficient)
      LeftBound32689.bound (LeftBound32689.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound32689.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32705 .coefficient)
      LeftAuthority32702.bound (LeftAuthority32702.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority32702.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound32689.bound, LeftAuthority32702.bound]
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32689.bound, LeftAuthority32702.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound32689.actual selector witness, LeftAuthority32702.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32706

namespace LeftBound32709
def owner : Owner := ⟨.program ⟨214⟩, ⟨16602⟩⟩
def transferEvent : Nat := 32709
def frameStart : Nat := 32650
def rule : BoundRule := .identity (.predecessor 0 32708 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32708 .coefficient)
      LeftBound32706.bound (LeftBound32706.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound32706.derived selector witness)

def rawBound : CoeffClass := LeftBound32706.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32706.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound32706.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound32709

namespace LeftBound32715
def owner : Owner := ⟨.program ⟨214⟩, ⟨16603⟩⟩
def transferEvent : Nat := 32715
def frameStart : Nat := 32650
def rule : BoundRule := .product (.predecessor 0 32713 .coefficient) (.predecessor 1 32714 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32713 .coefficient)
      LeftAuthority32711.bound (LeftAuthority32711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32712RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32711.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32711.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32714 .coefficient)
      LeftBound32709.bound (LeftBound32709.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32710RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32709.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32709.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority32711.bound LeftBound32709.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32711.bound, LeftBound32709.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority32711.actual selector witness) * (LeftBound32709.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32715

namespace LeftBound32723
def owner : Owner := ⟨.program ⟨214⟩, ⟨16604⟩⟩
def transferEvent : Nat := 32723
def frameStart : Nat := 32650
def rule : BoundRule := .sum [.predecessor 0 32721 .coefficient, .predecessor 1 32722 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32721 .coefficient)
      LeftAuthority32719.bound (LeftAuthority32719.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32720RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32719.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32719.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32722 .coefficient)
      LeftBound32715.bound (LeftBound32715.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32715.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32715.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority32719.bound, LeftBound32715.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32719.bound, LeftBound32715.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority32719.actual selector witness, LeftBound32715.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32723

namespace LeftBound32727
def owner : Owner := ⟨.program ⟨214⟩, ⟨29201⟩⟩
def transferEvent : Nat := 32727
def frameStart : Nat := 32650
def rule : BoundRule := .product (.predecessor 0 32725 .coefficient) (.predecessor 1 32726 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32725 .coefficient)
      LeftBound32723.bound (LeftBound32723.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32724RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32723.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32723.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32726 .coefficient)
      LeftAuthority32700.bound (LeftAuthority32700.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32701RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32700.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32700.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound32723.bound LeftAuthority32700.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32723.bound, LeftAuthority32700.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound32723.actual selector witness) * (LeftAuthority32700.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32727

namespace LeftBound32738
def owner : Owner := ⟨.program ⟨214⟩, ⟨17963⟩⟩
def transferEvent : Nat := 32738
def frameStart : Nat := 32650
def rule : BoundRule := .product (.predecessor 0 32736 .coefficient) (.predecessor 1 32737 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32736 .coefficient)
      LeftAuthority32711.bound (LeftAuthority32711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32712RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32711.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32711.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32737 .coefficient)
      LeftAuthority32734.bound (LeftAuthority32734.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32735RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32734.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32734.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority32711.bound LeftAuthority32734.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32711.bound, LeftAuthority32734.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority32711.actual selector witness) * (LeftAuthority32734.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32738

namespace LeftBound32746
def owner : Owner := ⟨.program ⟨214⟩, ⟨17964⟩⟩
def transferEvent : Nat := 32746
def frameStart : Nat := 32650
def rule : BoundRule := .sum [.predecessor 0 32744 .coefficient, .predecessor 1 32745 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32744 .coefficient)
      LeftAuthority32742.bound (LeftAuthority32742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32743RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32742.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32745 .coefficient)
      LeftBound32738.bound (LeftBound32738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32738.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32738.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority32742.bound, LeftBound32738.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32742.bound, LeftBound32738.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority32742.actual selector witness, LeftBound32738.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32746

namespace LeftBound32750
def owner : Owner := ⟨.program ⟨214⟩, ⟨29206⟩⟩
def transferEvent : Nat := 32750
def frameStart : Nat := 32650
def rule : BoundRule := .sum [.predecessor 0 32748 .coefficient, .predecessor 1 32749 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32748 .coefficient)
      LeftBound32746.bound (LeftBound32746.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32747RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32746.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32746.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32749 .coefficient)
      LeftBound32727.bound (LeftBound32727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32727.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32727.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound32746.bound, LeftBound32727.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32746.bound, LeftBound32727.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound32746.actual selector witness, LeftBound32727.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32750

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
