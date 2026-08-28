import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard383

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound56651
def owner : Owner := ⟨.program ⟨214⟩, ⟨13884⟩⟩
def transferEvent : Nat := 56651
def frameStart : Nat := 56601
def rule : BoundRule := .sum [.predecessor 0 56649 .coefficient, .predecessor 1 56650 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56649 .coefficient)
      LeftBound56634.bound (LeftBound56634.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound56634.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56650 .coefficient)
      LeftAuthority56647.bound (LeftAuthority56647.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority56647.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56634.bound, LeftAuthority56647.bound]
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56634.bound, LeftAuthority56647.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56634.actual selector witness, LeftAuthority56647.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56651

namespace LeftBound56654
def owner : Owner := ⟨.program ⟨214⟩, ⟨13885⟩⟩
def transferEvent : Nat := 56654
def frameStart : Nat := 56601
def rule : BoundRule := .identity (.predecessor 0 56653 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56653 .coefficient)
      LeftBound56651.bound (LeftBound56651.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound56651.derived selector witness)

def rawBound : CoeffClass := LeftBound56651.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56651.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound56651.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound56654

namespace LeftBound56660
def owner : Owner := ⟨.program ⟨214⟩, ⟨13886⟩⟩
def transferEvent : Nat := 56660
def frameStart : Nat := 56601
def rule : BoundRule := .product (.predecessor 0 56658 .coefficient) (.predecessor 1 56659 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56658 .coefficient)
      LeftAuthority56656.bound (LeftAuthority56656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56657RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56656.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56656.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56659 .coefficient)
      LeftBound56654.bound (LeftBound56654.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56654.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56654.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority56656.bound LeftBound56654.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56656.bound, LeftBound56654.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority56656.actual selector witness) * (LeftBound56654.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56660

namespace LeftBound56676
def owner : Owner := ⟨.program ⟨214⟩, ⟨7847⟩⟩
def transferEvent : Nat := 56676
def frameStart : Nat := 56601
def rule : BoundRule := .scale (.predecessor 0 56674 .coefficient) (.value (.predecessor 1 56675 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56674 .coefficient)
      LeftAuthority56672.bound (LeftAuthority56672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56672.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56672.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56675 .coefficient)
      LeftAuthority56663.bound (LeftAuthority56663.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority56663.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority56672.bound LeftAuthority56663.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56672.bound, LeftAuthority56663.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority56672.actual selector witness) * (LeftAuthority56663.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound56676

namespace LeftBound56679
def owner : Owner := ⟨.program ⟨214⟩, ⟨6794⟩⟩
def transferEvent : Nat := 56679
def frameStart : Nat := 56601
def rule : BoundRule := .identity (.predecessor 0 56678 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56678 .coefficient)
      LeftAuthority56666.bound (LeftAuthority56666.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56667RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56666.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56666.derived selector witness)

def rawBound : CoeffClass := LeftAuthority56666.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56666.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority56666.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound56679

namespace LeftBound56683
def owner : Owner := ⟨.program ⟨214⟩, ⟨7848⟩⟩
def transferEvent : Nat := 56683
def frameStart : Nat := 56601
def rule : BoundRule := .product (.predecessor 0 56681 .coefficient) (.predecessor 1 56682 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56681 .coefficient)
      LeftBound56679.bound (LeftBound56679.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56680RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56679.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56679.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56682 .coefficient)
      LeftBound56676.bound (LeftBound56676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56676.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56676.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56679.bound LeftBound56676.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56679.bound, LeftBound56676.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56679.actual selector witness) * (LeftBound56676.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56683

namespace LeftBound56688
def owner : Owner := ⟨.program ⟨214⟩, ⟨13887⟩⟩
def transferEvent : Nat := 56688
def frameStart : Nat := 56601
def rule : BoundRule := .sum [.predecessor 0 56686 .coefficient, .predecessor 1 56687 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56686 .coefficient)
      LeftBound56683.bound (LeftBound56683.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56685RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56683.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56683.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56687 .coefficient)
      LeftBound56660.bound (LeftBound56660.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56662RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56660.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56660.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56683.bound, LeftBound56660.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56683.bound, LeftBound56660.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56683.actual selector witness, LeftBound56660.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56688

namespace LeftBound56692
def owner : Owner := ⟨.program ⟨214⟩, ⟨25920⟩⟩
def transferEvent : Nat := 56692
def frameStart : Nat := 56601
def rule : BoundRule := .product (.predecessor 0 56690 .coefficient) (.predecessor 1 56691 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56690 .coefficient)
      LeftBound56688.bound (LeftBound56688.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56688.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56688.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56691 .coefficient)
      LeftAuthority56645.bound (LeftAuthority56645.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56646RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56645.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56645.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56688.bound LeftAuthority56645.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56688.bound, LeftAuthority56645.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56688.actual selector witness) * (LeftAuthority56645.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56692

namespace LeftBound56703
def owner : Owner := ⟨.program ⟨214⟩, ⟨15708⟩⟩
def transferEvent : Nat := 56703
def frameStart : Nat := 56601
def rule : BoundRule := .product (.predecessor 0 56701 .coefficient) (.predecessor 1 56702 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56701 .coefficient)
      LeftAuthority56656.bound (LeftAuthority56656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56657RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56656.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56656.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56702 .coefficient)
      LeftAuthority56699.bound (LeftAuthority56699.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56699.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56699.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority56656.bound LeftAuthority56699.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56656.bound, LeftAuthority56699.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority56656.actual selector witness) * (LeftAuthority56699.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56703

namespace LeftBound56711
def owner : Owner := ⟨.program ⟨214⟩, ⟨15709⟩⟩
def transferEvent : Nat := 56711
def frameStart : Nat := 56601
def rule : BoundRule := .sum [.predecessor 0 56709 .coefficient, .predecessor 1 56710 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56709 .coefficient)
      LeftAuthority56707.bound (LeftAuthority56707.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56708RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56707.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56707.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56710 .coefficient)
      LeftBound56703.bound (LeftBound56703.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56705RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56703.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56703.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority56707.bound, LeftBound56703.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56707.bound, LeftBound56703.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority56707.actual selector witness, LeftBound56703.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56711

namespace LeftBound56715
def owner : Owner := ⟨.program ⟨214⟩, ⟨25921⟩⟩
def transferEvent : Nat := 56715
def frameStart : Nat := 56601
def rule : BoundRule := .sum [.predecessor 0 56713 .coefficient, .predecessor 1 56714 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56713 .coefficient)
      LeftBound56711.bound (LeftBound56711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56712RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56711.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56711.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56714 .coefficient)
      LeftBound56692.bound (LeftBound56692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56697RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56692.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56711.bound, LeftBound56692.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56711.bound, LeftBound56692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56711.actual selector witness, LeftBound56692.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56715

namespace LeftBound56728
def owner : Owner := ⟨.program ⟨214⟩, ⟨25919⟩⟩
def transferEvent : Nat := 56728
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 56726 .coefficient, .predecessor 1 56727 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56726 .coefficient)
      LeftBound56549.bound (LeftBound56549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56549.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56549.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56727 .coefficient)
      LeftBound56532.bound (LeftBound56532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56539RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56532.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56532.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56549.bound, LeftBound56532.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56549.bound, LeftBound56532.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56549.actual selector witness, LeftBound56532.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56728

namespace LeftBound56731
def owner : Owner := ⟨.program ⟨214⟩, ⟨25919⟩⟩
def transferEvent : Nat := 56731
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 56725 .summary, .result 56539 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56725 .summary)
      LeftBound56551.bound (LeftBound56551.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19391⟩⟩) (rawTerms := some (Proof.Events221.exact56725RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56551.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56539 .summary)
      LeftBound56534.bound (LeftBound56534.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25918⟩⟩) (rawTerms := some (Proof.Events220.exact56539RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56534.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56551.bound, LeftBound56534.bound]
def bound : CoeffClass := .finite ⟨352042398396416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56551.bound, LeftBound56534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56551.actual selector witness, LeftBound56534.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56731

namespace LeftBound56735
def owner : Owner := ⟨.program ⟨214⟩, ⟨27447⟩⟩
def transferEvent : Nat := 56735
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 56733 .coefficient) (.predecessor 1 56734 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56733 .coefficient)
      LeftBound56728.bound (LeftBound56728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56728.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56734 .coefficient)
      LeftAuthority56454.bound (LeftAuthority56454.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56455RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56454.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56454.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56728.bound LeftAuthority56454.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56728.bound, LeftAuthority56454.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56728.actual selector witness) * (LeftAuthority56454.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56735

namespace LeftBound56736
def owner : Owner := ⟨.program ⟨214⟩, ⟨27447⟩⟩
def transferEvent : Nat := 56736
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27445⟩⟩]⟩ [⟨.result 56455 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56455 .coefficient)
      LeftAuthority56454.bound (LeftAuthority56454.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27445⟩⟩) (rawTerms := some (Proof.Events220.exact56455RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56454.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56454.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority56454.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56454.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority56454.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound56736

namespace LeftBound56737
def owner : Owner := ⟨.program ⟨214⟩, ⟨27447⟩⟩
def transferEvent : Nat := 56737
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 56732 .summary) (.transfer 56736) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56732 .summary)
      LeftBound56731.bound (LeftBound56731.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25919⟩⟩) (rawTerms := some (Proof.Events221.exact56732RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56731.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 56736)
      LeftBound56736.bound (LeftBound56736.actual selector witness) := by
  exact .transfer (LeftBound56736.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56731.bound LeftBound56736.bound
def bound : CoeffClass := .finite ⟨1292001234793221062656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56731.bound, LeftBound56736.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56731.actual selector witness) * (LeftBound56736.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56737

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
