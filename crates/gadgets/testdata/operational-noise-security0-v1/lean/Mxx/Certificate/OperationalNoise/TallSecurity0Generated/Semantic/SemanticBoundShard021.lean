import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound5818
def owner : Owner := ⟨.program ⟨214⟩, ⟨6664⟩⟩
def transferEvent : Nat := 5818
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 5816 .coefficient) (.value (.predecessor 1 5817 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5816 .coefficient)
      LeftAuthority5814.bound (LeftAuthority5814.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5814.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5814.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5817 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority5814.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5814.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5814.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound5818

namespace LeftBound5825
def owner : Owner := ⟨.program ⟨214⟩, ⟨7633⟩⟩
def transferEvent : Nat := 5825
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 5823 .coefficient) (.predecessor 1 5824 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5823 .coefficient)
      LeftAuthority5821.bound (LeftAuthority5821.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5822RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5821.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5821.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5824 .coefficient)
      LeftBound5818.bound (LeftBound5818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5819RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5818.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5818.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftAuthority5821.bound LeftBound5818.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5821.bound, LeftBound5818.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftAuthority5821.actual selector witness) * (LeftBound5818.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound5825

namespace LeftBound5830
def owner : Owner := ⟨.program ⟨214⟩, ⟨6607⟩⟩
def transferEvent : Nat := 5830
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 5828 .coefficient) (.predecessor 1 5829 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5828 .coefficient)
      LeftAuthority1.bound (LeftAuthority1.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact2RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5829 .coefficient)
      LeftAuthority702.bound (LeftAuthority702.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact703RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority702.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority702.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority1.bound LeftAuthority702.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1.bound, LeftAuthority702.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority1.actual selector witness) * (LeftAuthority702.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound5830

namespace LeftBound5838
def owner : Owner := ⟨.program ⟨214⟩, ⟨6672⟩⟩
def transferEvent : Nat := 5838
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 5836 .coefficient) (.value (.predecessor 1 5837 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5836 .coefficient)
      LeftAuthority5834.bound (LeftAuthority5834.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5835RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5834.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5834.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5837 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority5834.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5834.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5834.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound5838

namespace LeftBound5845
def owner : Owner := ⟨.program ⟨214⟩, ⟨7632⟩⟩
def transferEvent : Nat := 5845
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 5843 .coefficient) (.predecessor 1 5844 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5843 .coefficient)
      LeftAuthority5841.bound (LeftAuthority5841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5841.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5841.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5844 .coefficient)
      LeftBound5838.bound (LeftBound5838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5838.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5838.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftAuthority5841.bound LeftBound5838.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5841.bound, LeftBound5838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftAuthority5841.actual selector witness) * (LeftBound5838.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound5845

namespace LeftBound5850
def owner : Owner := ⟨.program ⟨214⟩, ⟨6611⟩⟩
def transferEvent : Nat := 5850
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 5848 .coefficient) (.predecessor 1 5849 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5848 .coefficient)
      LeftAuthority1.bound (LeftAuthority1.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact2RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5849 .coefficient)
      LeftAuthority712.bound (LeftAuthority712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority712.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority712.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority1.bound LeftAuthority712.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1.bound, LeftAuthority712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority1.actual selector witness) * (LeftAuthority712.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound5850

namespace LeftBound5858
def owner : Owner := ⟨.program ⟨214⟩, ⟨6680⟩⟩
def transferEvent : Nat := 5858
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 5856 .coefficient) (.value (.predecessor 1 5857 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5856 .coefficient)
      LeftAuthority5854.bound (LeftAuthority5854.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5855RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5854.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5854.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5857 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority5854.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5854.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5854.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound5858

namespace LeftBound5865
def owner : Owner := ⟨.program ⟨214⟩, ⟨7631⟩⟩
def transferEvent : Nat := 5865
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 5863 .coefficient) (.predecessor 1 5864 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5863 .coefficient)
      LeftAuthority5861.bound (LeftAuthority5861.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5862RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5861.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5861.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5864 .coefficient)
      LeftBound5858.bound (LeftBound5858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5858.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5858.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftAuthority5861.bound LeftBound5858.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5861.bound, LeftBound5858.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftAuthority5861.actual selector witness) * (LeftBound5858.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound5865

namespace LeftBound5872
def owner : Owner := ⟨.program ⟨214⟩, ⟨6760⟩⟩
def transferEvent : Nat := 5872
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 5871 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5871 .coefficient)
      LeftAuthority5869.bound (LeftAuthority5869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5869.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5869.derived selector witness)

def rawBound : CoeffClass := LeftAuthority5869.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5869.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority5869.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound5872

namespace LeftBound5876
def owner : Owner := ⟨.program ⟨214⟩, ⟨7650⟩⟩
def transferEvent : Nat := 5876
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5874 .coefficient, .predecessor 1 5875 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5874 .coefficient)
      LeftBound5872.bound (LeftBound5872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5872.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5875 .coefficient)
      LeftBound5872.bound (LeftBound5872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5872.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5872.bound, LeftBound5872.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5872.bound, LeftBound5872.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5872.actual selector witness, LeftBound5872.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5876

namespace LeftBound5881
def owner : Owner := ⟨.program ⟨214⟩, ⟨7651⟩⟩
def transferEvent : Nat := 5881
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5879 .coefficient, .predecessor 1 5880 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5879 .coefficient)
      LeftBound5876.bound (LeftBound5876.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5876.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5876.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5880 .coefficient)
      LeftBound5865.bound (LeftBound5865.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5867RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5865.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5865.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5876.bound, LeftBound5865.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5876.bound, LeftBound5865.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5876.actual selector witness, LeftBound5865.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5881

namespace LeftBound5885
def owner : Owner := ⟨.program ⟨214⟩, ⟨7652⟩⟩
def transferEvent : Nat := 5885
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5883 .coefficient, .predecessor 1 5884 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5883 .coefficient)
      LeftBound5881.bound (LeftBound5881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5882RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5881.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5881.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5884 .coefficient)
      LeftBound5845.bound (LeftBound5845.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5845.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5845.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5881.bound, LeftBound5845.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5881.bound, LeftBound5845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5881.actual selector witness, LeftBound5845.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5885

namespace LeftBound5889
def owner : Owner := ⟨.program ⟨214⟩, ⟨7653⟩⟩
def transferEvent : Nat := 5889
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5887 .coefficient, .predecessor 1 5888 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5887 .coefficient)
      LeftBound5885.bound (LeftBound5885.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5886RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5885.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5885.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5888 .coefficient)
      LeftBound5825.bound (LeftBound5825.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5827RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5825.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5825.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5885.bound, LeftBound5825.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5885.bound, LeftBound5825.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5885.actual selector witness, LeftBound5825.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5889

namespace LeftBound5893
def owner : Owner := ⟨.program ⟨214⟩, ⟨7654⟩⟩
def transferEvent : Nat := 5893
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5891 .coefficient, .predecessor 1 5892 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5891 .coefficient)
      LeftBound5889.bound (LeftBound5889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5890RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5889.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5889.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5892 .coefficient)
      LeftBound5805.bound (LeftBound5805.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5807RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5805.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5805.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5889.bound, LeftBound5805.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5889.bound, LeftBound5805.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5889.actual selector witness, LeftBound5805.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5893

namespace LeftBound5897
def owner : Owner := ⟨.program ⟨214⟩, ⟨7655⟩⟩
def transferEvent : Nat := 5897
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5895 .coefficient, .predecessor 1 5896 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5895 .coefficient)
      LeftBound5893.bound (LeftBound5893.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5894RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5893.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5893.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5896 .coefficient)
      LeftBound5785.bound (LeftBound5785.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5787RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5785.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5785.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5893.bound, LeftBound5785.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5893.bound, LeftBound5785.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5893.actual selector witness, LeftBound5785.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5897

namespace LeftBound5901
def owner : Owner := ⟨.program ⟨214⟩, ⟨7656⟩⟩
def transferEvent : Nat := 5901
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 5899 .coefficient, .predecessor 1 5900 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5899 .coefficient)
      LeftBound5897.bound (LeftBound5897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5897.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5900 .coefficient)
      LeftBound5765.bound (LeftBound5765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5765.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5765.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5897.bound, LeftBound5765.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5897.bound, LeftBound5765.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5897.actual selector witness, LeftBound5765.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound5901

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
