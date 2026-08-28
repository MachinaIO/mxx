import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound87691
def owner : Owner := ⟨.program ⟨214⟩, ⟨19098⟩⟩
def transferEvent : Nat := 87691
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 87689 .coefficient) (.value (.predecessor 1 87690 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87689 .coefficient)
      LeftAuthority87687.bound (LeftAuthority87687.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87688RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87687.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87687.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87690 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority87687.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87687.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority87687.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound87691

namespace LeftBound87695
def owner : Owner := ⟨.program ⟨214⟩, ⟨19099⟩⟩
def transferEvent : Nat := 87695
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 87693 .coefficient) (.predecessor 1 87694 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87693 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87694 .coefficient)
      LeftBound87691.bound (LeftBound87691.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87692RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87691.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87691.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound87691.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound87691.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound87691.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87695

namespace LeftBound87696
def owner : Owner := ⟨.program ⟨214⟩, ⟨19099⟩⟩
def transferEvent : Nat := 87696
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19096⟩⟩]⟩ [⟨.result 87688 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87688 .coefficient)
      LeftAuthority87687.bound (LeftAuthority87687.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19096⟩⟩) (rawTerms := some (Proof.Events342.exact87688RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87687.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87687.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority87687.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87687.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority87687.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound87696

namespace LeftBound87697
def owner : Owner := ⟨.program ⟨214⟩, ⟨19099⟩⟩
def transferEvent : Nat := 87697
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 87696) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 87696)
      LeftBound87696.bound (LeftBound87696.actual selector witness) := by
  exact .transfer (LeftBound87696.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound87696.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound87696.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound87696.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87697

namespace LeftBound87776
def owner : Owner := ⟨.program ⟨214⟩, ⟨10677⟩⟩
def transferEvent : Nat := 87776
def frameStart : Nat := 87747
def rule : BoundRule := .product (.predecessor 0 87774 .coefficient) (.predecessor 1 87775 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87774 .coefficient)
      LeftAuthority87772.bound (LeftAuthority87772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87773RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87772.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87772.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87775 .coefficient)
      LeftAuthority87769.bound (LeftAuthority87769.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87770RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87769.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87769.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority87772.bound LeftAuthority87769.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87772.bound, LeftAuthority87769.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority87772.actual selector witness) * (LeftAuthority87769.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87776

namespace LeftBound87780
def owner : Owner := ⟨.program ⟨214⟩, ⟨10678⟩⟩
def transferEvent : Nat := 87780
def frameStart : Nat := 87747
def rule : BoundRule := .identity (.predecessor 0 87779 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87779 .coefficient)
      LeftBound87776.bound (LeftBound87776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87776.derived selector witness)

def rawBound : CoeffClass := LeftBound87776.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound87776.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound87780

namespace LeftBound87797
def owner : Owner := ⟨.program ⟨214⟩, ⟨10772⟩⟩
def transferEvent : Nat := 87797
def frameStart : Nat := 87747
def rule : BoundRule := .sum [.predecessor 0 87795 .coefficient, .predecessor 1 87796 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87795 .coefficient)
      LeftBound87780.bound (LeftBound87780.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound87780.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87796 .coefficient)
      LeftAuthority87793.bound (LeftAuthority87793.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority87793.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87780.bound, LeftAuthority87793.bound]
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87780.bound, LeftAuthority87793.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87780.actual selector witness, LeftAuthority87793.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87797

namespace LeftBound87800
def owner : Owner := ⟨.program ⟨214⟩, ⟨10773⟩⟩
def transferEvent : Nat := 87800
def frameStart : Nat := 87747
def rule : BoundRule := .identity (.predecessor 0 87799 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87799 .coefficient)
      LeftBound87797.bound (LeftBound87797.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound87797.derived selector witness)

def rawBound : CoeffClass := LeftBound87797.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87797.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound87797.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound87800

namespace LeftBound87806
def owner : Owner := ⟨.program ⟨214⟩, ⟨10774⟩⟩
def transferEvent : Nat := 87806
def frameStart : Nat := 87747
def rule : BoundRule := .product (.predecessor 0 87804 .coefficient) (.predecessor 1 87805 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87804 .coefficient)
      LeftAuthority87802.bound (LeftAuthority87802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87803RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87802.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87802.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87805 .coefficient)
      LeftBound87800.bound (LeftBound87800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87801RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87800.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87800.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority87802.bound LeftBound87800.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87802.bound, LeftBound87800.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority87802.actual selector witness) * (LeftBound87800.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87806

namespace LeftBound87820
def owner : Owner := ⟨.program ⟨214⟩, ⟨7835⟩⟩
def transferEvent : Nat := 87820
def frameStart : Nat := 87747
def rule : BoundRule := .scale (.predecessor 0 87818 .coefficient) (.value (.predecessor 1 87819 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87818 .coefficient)
      LeftAuthority87816.bound (LeftAuthority87816.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87817RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87816.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87816.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87819 .coefficient)
      LeftAuthority87750.bound (LeftAuthority87750.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority87750.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority87816.bound LeftAuthority87750.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87816.bound, LeftAuthority87750.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority87816.actual selector witness) * (LeftAuthority87750.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound87820

namespace LeftBound87823
def owner : Owner := ⟨.program ⟨214⟩, ⟨6782⟩⟩
def transferEvent : Nat := 87823
def frameStart : Nat := 87747
def rule : BoundRule := .identity (.predecessor 0 87822 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87822 .coefficient)
      LeftAuthority87810.bound (LeftAuthority87810.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87811RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87810.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87810.derived selector witness)

def rawBound : CoeffClass := LeftAuthority87810.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority87810.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound87823

namespace LeftBound87827
def owner : Owner := ⟨.program ⟨214⟩, ⟨7836⟩⟩
def transferEvent : Nat := 87827
def frameStart : Nat := 87747
def rule : BoundRule := .product (.predecessor 0 87825 .coefficient) (.predecessor 1 87826 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87825 .coefficient)
      LeftBound87823.bound (LeftBound87823.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87824RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87823.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87823.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87826 .coefficient)
      LeftBound87820.bound (LeftBound87820.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87821RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87820.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87820.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound87823.bound LeftBound87820.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87823.bound, LeftBound87820.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound87823.actual selector witness) * (LeftBound87820.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87827

namespace LeftBound87832
def owner : Owner := ⟨.program ⟨214⟩, ⟨10775⟩⟩
def transferEvent : Nat := 87832
def frameStart : Nat := 87747
def rule : BoundRule := .sum [.predecessor 0 87830 .coefficient, .predecessor 1 87831 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87830 .coefficient)
      LeftBound87827.bound (LeftBound87827.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87829RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87827.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87827.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87831 .coefficient)
      LeftBound87806.bound (LeftBound87806.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87808RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87806.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87806.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87827.bound, LeftBound87806.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87827.bound, LeftBound87806.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87827.actual selector witness, LeftBound87806.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87832

namespace LeftBound87836
def owner : Owner := ⟨.program ⟨214⟩, ⟨24991⟩⟩
def transferEvent : Nat := 87836
def frameStart : Nat := 87747
def rule : BoundRule := .product (.predecessor 0 87834 .coefficient) (.predecessor 1 87835 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87834 .coefficient)
      LeftBound87832.bound (LeftBound87832.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87833RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87832.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87832.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87835 .coefficient)
      LeftAuthority87791.bound (LeftAuthority87791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87792RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87791.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87791.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound87832.bound LeftAuthority87791.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87832.bound, LeftAuthority87791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound87832.actual selector witness) * (LeftAuthority87791.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87836

namespace LeftBound87847
def owner : Owner := ⟨.program ⟨214⟩, ⟨14955⟩⟩
def transferEvent : Nat := 87847
def frameStart : Nat := 87747
def rule : BoundRule := .product (.predecessor 0 87845 .coefficient) (.predecessor 1 87846 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87845 .coefficient)
      LeftAuthority87802.bound (LeftAuthority87802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87803RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87802.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87802.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87846 .coefficient)
      LeftAuthority87843.bound (LeftAuthority87843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87844RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87843.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87843.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority87802.bound LeftAuthority87843.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87802.bound, LeftAuthority87843.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority87802.actual selector witness) * (LeftAuthority87843.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87847

namespace LeftBound87855
def owner : Owner := ⟨.program ⟨214⟩, ⟨14956⟩⟩
def transferEvent : Nat := 87855
def frameStart : Nat := 87747
def rule : BoundRule := .sum [.predecessor 0 87853 .coefficient, .predecessor 1 87854 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87853 .coefficient)
      LeftAuthority87851.bound (LeftAuthority87851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87851.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87851.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87854 .coefficient)
      LeftBound87847.bound (LeftBound87847.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87849RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87847.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87847.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority87851.bound, LeftBound87847.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87851.bound, LeftBound87847.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority87851.actual selector witness, LeftBound87847.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87855

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
