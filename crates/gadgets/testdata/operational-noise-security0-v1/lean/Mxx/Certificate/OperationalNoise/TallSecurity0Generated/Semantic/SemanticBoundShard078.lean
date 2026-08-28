import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard077

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound12661
def owner : Owner := ⟨.program ⟨214⟩, ⟨13811⟩⟩
def transferEvent : Nat := 12661
def frameStart : Nat := 12628
def rule : BoundRule := .identity (.predecessor 0 12660 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12660 .coefficient)
      LeftBound12657.bound (LeftBound12657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12657.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12657.derived selector witness)

def rawBound : CoeffClass := LeftBound12657.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12657.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound12657.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound12661

namespace LeftBound12678
def owner : Owner := ⟨.program ⟨214⟩, ⟨13896⟩⟩
def transferEvent : Nat := 12678
def frameStart : Nat := 12628
def rule : BoundRule := .sum [.predecessor 0 12676 .coefficient, .predecessor 1 12677 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12676 .coefficient)
      LeftBound12661.bound (LeftBound12661.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound12661.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12677 .coefficient)
      LeftAuthority12674.bound (LeftAuthority12674.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority12674.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12661.bound, LeftAuthority12674.bound]
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12661.bound, LeftAuthority12674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound12661.actual selector witness, LeftAuthority12674.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12678

namespace LeftBound12681
def owner : Owner := ⟨.program ⟨214⟩, ⟨13897⟩⟩
def transferEvent : Nat := 12681
def frameStart : Nat := 12628
def rule : BoundRule := .identity (.predecessor 0 12680 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12680 .coefficient)
      LeftBound12678.bound (LeftBound12678.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound12678.derived selector witness)

def rawBound : CoeffClass := LeftBound12678.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound12678.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound12681

namespace LeftBound12687
def owner : Owner := ⟨.program ⟨214⟩, ⟨13898⟩⟩
def transferEvent : Nat := 12687
def frameStart : Nat := 12628
def rule : BoundRule := .product (.predecessor 0 12685 .coefficient) (.predecessor 1 12686 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12685 .coefficient)
      LeftAuthority12683.bound (LeftAuthority12683.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12684RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12683.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12683.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12686 .coefficient)
      LeftBound12681.bound (LeftBound12681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12682RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12681.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12681.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority12683.bound LeftBound12681.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12683.bound, LeftBound12681.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority12683.actual selector witness) * (LeftBound12681.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12687

namespace LeftBound12703
def owner : Owner := ⟨.program ⟨214⟩, ⟨7847⟩⟩
def transferEvent : Nat := 12703
def frameStart : Nat := 12628
def rule : BoundRule := .scale (.predecessor 0 12701 .coefficient) (.value (.predecessor 1 12702 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12701 .coefficient)
      LeftAuthority12699.bound (LeftAuthority12699.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12699.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12699.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12702 .coefficient)
      LeftAuthority12690.bound (LeftAuthority12690.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority12690.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority12699.bound LeftAuthority12690.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12699.bound, LeftAuthority12690.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority12699.actual selector witness) * (LeftAuthority12690.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound12703

namespace LeftBound12706
def owner : Owner := ⟨.program ⟨214⟩, ⟨6794⟩⟩
def transferEvent : Nat := 12706
def frameStart : Nat := 12628
def rule : BoundRule := .identity (.predecessor 0 12705 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12705 .coefficient)
      LeftAuthority12693.bound (LeftAuthority12693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12694RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12693.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12693.derived selector witness)

def rawBound : CoeffClass := LeftAuthority12693.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12693.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority12693.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound12706

namespace LeftBound12710
def owner : Owner := ⟨.program ⟨214⟩, ⟨7848⟩⟩
def transferEvent : Nat := 12710
def frameStart : Nat := 12628
def rule : BoundRule := .product (.predecessor 0 12708 .coefficient) (.predecessor 1 12709 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12708 .coefficient)
      LeftBound12706.bound (LeftBound12706.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12707RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12706.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12706.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12709 .coefficient)
      LeftBound12703.bound (LeftBound12703.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12704RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12703.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12703.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound12706.bound LeftBound12703.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12706.bound, LeftBound12703.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound12706.actual selector witness) * (LeftBound12703.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12710

namespace LeftBound12715
def owner : Owner := ⟨.program ⟨214⟩, ⟨13899⟩⟩
def transferEvent : Nat := 12715
def frameStart : Nat := 12628
def rule : BoundRule := .sum [.predecessor 0 12713 .coefficient, .predecessor 1 12714 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12713 .coefficient)
      LeftBound12710.bound (LeftBound12710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12712RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12710.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12710.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12714 .coefficient)
      LeftBound12687.bound (LeftBound12687.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12687.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12687.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12710.bound, LeftBound12687.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12710.bound, LeftBound12687.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound12710.actual selector witness, LeftBound12687.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12715

namespace LeftBound12719
def owner : Owner := ⟨.program ⟨214⟩, ⟨25935⟩⟩
def transferEvent : Nat := 12719
def frameStart : Nat := 12628
def rule : BoundRule := .product (.predecessor 0 12717 .coefficient) (.predecessor 1 12718 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12717 .coefficient)
      LeftBound12715.bound (LeftBound12715.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12716RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12715.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12715.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12718 .coefficient)
      LeftAuthority12672.bound (LeftAuthority12672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12672.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12672.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound12715.bound LeftAuthority12672.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12715.bound, LeftAuthority12672.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound12715.actual selector witness) * (LeftAuthority12672.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12719

namespace LeftBound12730
def owner : Owner := ⟨.program ⟨214⟩, ⟨15720⟩⟩
def transferEvent : Nat := 12730
def frameStart : Nat := 12628
def rule : BoundRule := .product (.predecessor 0 12728 .coefficient) (.predecessor 1 12729 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12728 .coefficient)
      LeftAuthority12683.bound (LeftAuthority12683.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12684RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12683.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12683.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12729 .coefficient)
      LeftAuthority12726.bound (LeftAuthority12726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12726.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12726.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority12683.bound LeftAuthority12726.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12683.bound, LeftAuthority12726.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority12683.actual selector witness) * (LeftAuthority12726.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12730

namespace LeftBound12738
def owner : Owner := ⟨.program ⟨214⟩, ⟨15721⟩⟩
def transferEvent : Nat := 12738
def frameStart : Nat := 12628
def rule : BoundRule := .sum [.predecessor 0 12736 .coefficient, .predecessor 1 12737 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12736 .coefficient)
      LeftAuthority12734.bound (LeftAuthority12734.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12735RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12734.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12734.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12737 .coefficient)
      LeftBound12730.bound (LeftBound12730.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12730.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12730.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority12734.bound, LeftBound12730.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12734.bound, LeftBound12730.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority12734.actual selector witness, LeftBound12730.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12738

namespace LeftBound12742
def owner : Owner := ⟨.program ⟨214⟩, ⟨25936⟩⟩
def transferEvent : Nat := 12742
def frameStart : Nat := 12628
def rule : BoundRule := .sum [.predecessor 0 12740 .coefficient, .predecessor 1 12741 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12740 .coefficient)
      LeftBound12738.bound (LeftBound12738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12739RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12738.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12738.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12741 .coefficient)
      LeftBound12719.bound (LeftBound12719.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12724RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12719.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12719.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12738.bound, LeftBound12719.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12738.bound, LeftBound12719.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound12738.actual selector witness, LeftBound12719.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12742

namespace LeftBound12755
def owner : Owner := ⟨.program ⟨214⟩, ⟨25934⟩⟩
def transferEvent : Nat := 12755
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 12753 .coefficient, .predecessor 1 12754 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12753 .coefficient)
      LeftBound12576.bound (LeftBound12576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12752RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12576.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12576.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12754 .coefficient)
      LeftBound12559.bound (LeftBound12559.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12566RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12559.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12559.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12576.bound, LeftBound12559.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12576.bound, LeftBound12559.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound12576.actual selector witness, LeftBound12559.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12755

namespace LeftBound12758
def owner : Owner := ⟨.program ⟨214⟩, ⟨25934⟩⟩
def transferEvent : Nat := 12758
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 12752 .summary, .result 12566 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12752 .summary)
      LeftBound12578.bound (LeftBound12578.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19403⟩⟩) (rawTerms := some (Proof.Events049.exact12752RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound12578.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12566 .summary)
      LeftBound12561.bound (LeftBound12561.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25933⟩⟩) (rawTerms := some (Proof.Events049.exact12566RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound12561.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12578.bound, LeftBound12561.bound]
def bound : CoeffClass := .finite ⟨352042398396416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12578.bound, LeftBound12561.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound12578.actual selector witness, LeftBound12561.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12758

namespace LeftBound12762
def owner : Owner := ⟨.program ⟨214⟩, ⟨27486⟩⟩
def transferEvent : Nat := 12762
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 12760 .coefficient) (.predecessor 1 12761 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12760 .coefficient)
      LeftBound12755.bound (LeftBound12755.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12759RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12755.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12755.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12761 .coefficient)
      LeftAuthority12462.bound (LeftAuthority12462.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12462.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12462.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound12755.bound LeftAuthority12462.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12755.bound, LeftAuthority12462.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound12755.actual selector witness) * (LeftAuthority12462.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12762

namespace LeftBound12763
def owner : Owner := ⟨.program ⟨214⟩, ⟨27486⟩⟩
def transferEvent : Nat := 12763
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27484⟩⟩]⟩ [⟨.result 12463 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12463 .coefficient)
      LeftAuthority12462.bound (LeftAuthority12462.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27484⟩⟩) (rawTerms := some (Proof.Events048.exact12463RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12462.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12462.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority12462.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12462.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority12462.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound12763

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
