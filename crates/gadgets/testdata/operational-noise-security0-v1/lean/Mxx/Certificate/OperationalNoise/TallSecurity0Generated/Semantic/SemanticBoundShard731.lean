import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard730

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound106627
def owner : Owner := ⟨.program ⟨214⟩, ⟨20600⟩⟩
def transferEvent : Nat := 106627
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20597⟩⟩]⟩ [⟨.result 106619 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106619 .coefficient)
      LeftAuthority106618.bound (LeftAuthority106618.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20597⟩⟩) (rawTerms := some (Proof.Events416.exact106619RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106618.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106618.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority106618.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106618.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority106618.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound106627

namespace LeftBound106628
def owner : Owner := ⟨.program ⟨214⟩, ⟨20600⟩⟩
def transferEvent : Nat := 106628
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 106627) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 106627)
      LeftBound106627.bound (LeftBound106627.actual selector witness) := by
  exact .transfer (LeftBound106627.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound106627.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound106627.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound106627.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106628

namespace LeftBound106699
def owner : Owner := ⟨.program ⟨214⟩, ⟨15105⟩⟩
def transferEvent : Nat := 106699
def frameStart : Nat := 106672
def rule : BoundRule := .identity (.predecessor 0 106698 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106698 .coefficient)
      LeftAuthority106696.bound (LeftAuthority106696.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106697RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106696.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106696.derived selector witness)

def rawBound : CoeffClass := LeftAuthority106696.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106696.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority106696.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound106699

namespace LeftBound106716
def owner : Owner := ⟨.program ⟨214⟩, ⟨15146⟩⟩
def transferEvent : Nat := 106716
def frameStart : Nat := 106672
def rule : BoundRule := .sum [.predecessor 0 106714 .coefficient, .predecessor 1 106715 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106714 .coefficient)
      LeftBound106699.bound (LeftBound106699.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound106699.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106715 .coefficient)
      LeftAuthority106712.bound (LeftAuthority106712.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority106712.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106699.bound, LeftAuthority106712.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106699.bound, LeftAuthority106712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound106699.actual selector witness, LeftAuthority106712.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106716

namespace LeftBound106719
def owner : Owner := ⟨.program ⟨214⟩, ⟨15147⟩⟩
def transferEvent : Nat := 106719
def frameStart : Nat := 106672
def rule : BoundRule := .identity (.predecessor 0 106718 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106718 .coefficient)
      LeftBound106716.bound (LeftBound106716.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound106716.derived selector witness)

def rawBound : CoeffClass := LeftBound106716.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106716.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound106716.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound106719

namespace LeftBound106725
def owner : Owner := ⟨.program ⟨214⟩, ⟨15148⟩⟩
def transferEvent : Nat := 106725
def frameStart : Nat := 106672
def rule : BoundRule := .product (.predecessor 0 106723 .coefficient) (.predecessor 1 106724 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106723 .coefficient)
      LeftAuthority106721.bound (LeftAuthority106721.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106722RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106721.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106721.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106724 .coefficient)
      LeftBound106719.bound (LeftBound106719.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106720RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106719.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106719.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority106721.bound LeftBound106719.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106721.bound, LeftBound106719.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority106721.actual selector witness) * (LeftBound106719.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106725

namespace LeftBound106733
def owner : Owner := ⟨.program ⟨214⟩, ⟨15149⟩⟩
def transferEvent : Nat := 106733
def frameStart : Nat := 106672
def rule : BoundRule := .sum [.predecessor 0 106731 .coefficient, .predecessor 1 106732 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106731 .coefficient)
      LeftAuthority106729.bound (LeftAuthority106729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106730RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106729.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106732 .coefficient)
      LeftBound106725.bound (LeftBound106725.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106725.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106725.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority106729.bound, LeftBound106725.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106729.bound, LeftBound106725.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority106729.actual selector witness, LeftBound106725.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106733

namespace LeftBound106737
def owner : Owner := ⟨.program ⟨214⟩, ⟨26740⟩⟩
def transferEvent : Nat := 106737
def frameStart : Nat := 106672
def rule : BoundRule := .product (.predecessor 0 106735 .coefficient) (.predecessor 1 106736 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106735 .coefficient)
      LeftBound106733.bound (LeftBound106733.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106734RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106733.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106733.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106736 .coefficient)
      LeftAuthority106710.bound (LeftAuthority106710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106711RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106710.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106710.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound106733.bound LeftAuthority106710.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106733.bound, LeftAuthority106710.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound106733.actual selector witness) * (LeftAuthority106710.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106737

namespace LeftBound106748
def owner : Owner := ⟨.program ⟨214⟩, ⟨15198⟩⟩
def transferEvent : Nat := 106748
def frameStart : Nat := 106672
def rule : BoundRule := .product (.predecessor 0 106746 .coefficient) (.predecessor 1 106747 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106746 .coefficient)
      LeftAuthority106721.bound (LeftAuthority106721.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106722RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106721.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106721.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106747 .coefficient)
      LeftAuthority106744.bound (LeftAuthority106744.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106744.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106744.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority106721.bound LeftAuthority106744.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106721.bound, LeftAuthority106744.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority106721.actual selector witness) * (LeftAuthority106744.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106748

namespace LeftBound106756
def owner : Owner := ⟨.program ⟨214⟩, ⟨15199⟩⟩
def transferEvent : Nat := 106756
def frameStart : Nat := 106672
def rule : BoundRule := .sum [.predecessor 0 106754 .coefficient, .predecessor 1 106755 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106754 .coefficient)
      LeftAuthority106752.bound (LeftAuthority106752.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106753RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106752.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106752.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106755 .coefficient)
      LeftBound106748.bound (LeftBound106748.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106748.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106748.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority106752.bound, LeftBound106748.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106752.bound, LeftBound106748.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority106752.actual selector witness, LeftBound106748.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106756

namespace LeftBound106760
def owner : Owner := ⟨.program ⟨214⟩, ⟨26745⟩⟩
def transferEvent : Nat := 106760
def frameStart : Nat := 106672
def rule : BoundRule := .sum [.predecessor 0 106758 .coefficient, .predecessor 1 106759 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106758 .coefficient)
      LeftBound106756.bound (LeftBound106756.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106757RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106756.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106756.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106759 .coefficient)
      LeftBound106737.bound (LeftBound106737.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106742RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106737.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106737.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106756.bound, LeftBound106737.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106756.bound, LeftBound106737.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound106756.actual selector witness, LeftBound106737.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106760

namespace LeftBound106773
def owner : Owner := ⟨.program ⟨214⟩, ⟨26742⟩⟩
def transferEvent : Nat := 106773
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 106771 .coefficient, .predecessor 1 106772 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106771 .coefficient)
      LeftBound106626.bound (LeftBound106626.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106770RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106626.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106626.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106772 .coefficient)
      LeftBound106609.bound (LeftBound106609.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106609.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106609.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106626.bound, LeftBound106609.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106626.bound, LeftBound106609.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound106626.actual selector witness, LeftBound106609.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106773

namespace LeftBound106776
def owner : Owner := ⟨.program ⟨214⟩, ⟨26742⟩⟩
def transferEvent : Nat := 106776
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 106770 .summary, .result 106616 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106770 .summary)
      LeftBound106628.bound (LeftBound106628.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20600⟩⟩) (rawTerms := some (Proof.Events417.exact106770RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106628.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106616 .summary)
      LeftBound106611.bound (LeftBound106611.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26741⟩⟩) (rawTerms := some (Proof.Events416.exact106616RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106611.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106628.bound, LeftBound106611.bound]
def bound : CoeffClass := .finite ⟨1291911586824442228736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106628.bound, LeftBound106611.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound106628.actual selector witness, LeftBound106611.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106776

namespace LeftBound106780
def owner : Owner := ⟨.program ⟨214⟩, ⟨26743⟩⟩
def transferEvent : Nat := 106780
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 106778 .coefficient) (.predecessor 1 106779 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106778 .coefficient)
      LeftBound106773.bound (LeftBound106773.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106777RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106773.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106773.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106779 .coefficient)
      LeftBound5818.bound (LeftBound5818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5819RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5818.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5818.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound106773.bound LeftBound5818.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106773.bound, LeftBound5818.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound106773.actual selector witness) * (LeftBound5818.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106780

namespace LeftBound106781
def owner : Owner := ⟨.program ⟨214⟩, ⟨26743⟩⟩
def transferEvent : Nat := 106781
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩ [⟨.result 5815 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5815 .coefficient)
      LeftAuthority5814.bound (LeftAuthority5814.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6663⟩⟩) (rawTerms := some (Proof.Events022.exact5815RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5814.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5814.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5814.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5814.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5814.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound106781

namespace LeftBound106782
def owner : Owner := ⟨.program ⟨214⟩, ⟨26743⟩⟩
def transferEvent : Nat := 106782
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 106777 .summary) (.transfer 106781) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106777 .summary)
      LeftBound106776.bound (LeftBound106776.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26742⟩⟩) (rawTerms := some (Proof.Events417.exact106777RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106776.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 106781)
      LeftBound106781.bound (LeftBound106781.actual selector witness) := by
  exact .transfer (LeftBound106781.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound106776.bound LeftBound106781.bound
def bound : CoeffClass := .finite ⟨4741336194231092170536779776, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106776.bound, LeftBound106781.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound106776.actual selector witness) * (LeftBound106781.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106782

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
