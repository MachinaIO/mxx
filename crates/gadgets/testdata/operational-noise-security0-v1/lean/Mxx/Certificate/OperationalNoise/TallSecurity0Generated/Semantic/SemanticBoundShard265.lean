import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard263
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard264

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound39676
def owner : Owner := ⟨.program ⟨214⟩, ⟨16273⟩⟩
def transferEvent : Nat := 39676
def frameStart : Nat := 39566
def rule : BoundRule := .sum [.predecessor 0 39674 .coefficient, .predecessor 1 39675 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39674 .coefficient)
      LeftAuthority39672.bound (LeftAuthority39672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39672.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39672.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39675 .coefficient)
      LeftBound39668.bound (LeftBound39668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39668.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority39672.bound, LeftBound39668.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39672.bound, LeftBound39668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority39672.actual selector witness, LeftBound39668.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39676

namespace LeftBound39680
def owner : Owner := ⟨.program ⟨214⟩, ⟨25156⟩⟩
def transferEvent : Nat := 39680
def frameStart : Nat := 39566
def rule : BoundRule := .sum [.predecessor 0 39678 .coefficient, .predecessor 1 39679 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39678 .coefficient)
      LeftBound39676.bound (LeftBound39676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39676.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39676.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39679 .coefficient)
      LeftBound39657.bound (LeftBound39657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39662RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39657.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39657.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39676.bound, LeftBound39657.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39676.bound, LeftBound39657.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39676.actual selector witness, LeftBound39657.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39680

namespace LeftBound39693
def owner : Owner := ⟨.program ⟨214⟩, ⟨25154⟩⟩
def transferEvent : Nat := 39693
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 39691 .coefficient, .predecessor 1 39692 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39691 .coefficient)
      LeftBound39514.bound (LeftBound39514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39690RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39514.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39692 .coefficient)
      LeftBound39497.bound (LeftBound39497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39497.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39497.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39514.bound, LeftBound39497.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39514.bound, LeftBound39497.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39514.actual selector witness, LeftBound39497.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39693

namespace LeftBound39696
def owner : Owner := ⟨.program ⟨214⟩, ⟨25154⟩⟩
def transferEvent : Nat := 39696
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 39690 .summary, .result 39504 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39690 .summary)
      LeftBound39516.bound (LeftBound39516.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19755⟩⟩) (rawTerms := some (Proof.Events155.exact39690RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39516.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39504 .summary)
      LeftBound39499.bound (LeftBound39499.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25153⟩⟩) (rawTerms := some (Proof.Events154.exact39504RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39499.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39516.bound, LeftBound39499.bound]
def bound : CoeffClass := .finite ⟨352097360556032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39516.bound, LeftBound39499.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39516.actual selector witness, LeftBound39499.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39696

namespace LeftBound39700
def owner : Owner := ⟨.program ⟨214⟩, ⟨28545⟩⟩
def transferEvent : Nat := 39700
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 39698 .coefficient) (.predecessor 1 39699 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39698 .coefficient)
      LeftBound39693.bound (LeftBound39693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39697RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39693.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39693.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39699 .coefficient)
      LeftAuthority39419.bound (LeftAuthority39419.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39419.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39419.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound39693.bound LeftAuthority39419.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39693.bound, LeftAuthority39419.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound39693.actual selector witness) * (LeftAuthority39419.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39700

namespace LeftBound39701
def owner : Owner := ⟨.program ⟨214⟩, ⟨28545⟩⟩
def transferEvent : Nat := 39701
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28543⟩⟩]⟩ [⟨.result 39420 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39420 .coefficient)
      LeftAuthority39419.bound (LeftAuthority39419.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28543⟩⟩) (rawTerms := some (Proof.Events153.exact39420RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39419.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39419.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority39419.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39419.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority39419.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound39701

namespace LeftBound39702
def owner : Owner := ⟨.program ⟨214⟩, ⟨28545⟩⟩
def transferEvent : Nat := 39702
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 39697 .summary) (.transfer 39701) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39697 .summary)
      LeftBound39696.bound (LeftBound39696.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25154⟩⟩) (rawTerms := some (Proof.Events155.exact39697RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39696.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 39701)
      LeftBound39701.bound (LeftBound39701.actual selector witness) := by
  exact .transfer (LeftBound39701.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound39696.bound LeftBound39701.bound
def bound : CoeffClass := .finite ⟨1292202946798406336512, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39696.bound, LeftBound39701.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound39696.actual selector witness) * (LeftBound39701.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39702

namespace LeftBound39713
def owner : Owner := ⟨.program ⟨214⟩, ⟨21842⟩⟩
def transferEvent : Nat := 39713
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 39711 .coefficient) (.value (.predecessor 1 39712 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39711 .coefficient)
      LeftAuthority39709.bound (LeftAuthority39709.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39710RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39709.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39709.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39712 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority39709.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39709.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority39709.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound39713

namespace LeftBound39717
def owner : Owner := ⟨.program ⟨214⟩, ⟨21843⟩⟩
def transferEvent : Nat := 39717
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 39715 .coefficient) (.predecessor 1 39716 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39715 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39716 .coefficient)
      LeftBound39713.bound (LeftBound39713.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39714RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39713.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39713.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound39713.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound39713.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound39713.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39717

namespace LeftBound39718
def owner : Owner := ⟨.program ⟨214⟩, ⟨21843⟩⟩
def transferEvent : Nat := 39718
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21840⟩⟩]⟩ [⟨.result 39710 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39710 .coefficient)
      LeftAuthority39709.bound (LeftAuthority39709.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21840⟩⟩) (rawTerms := some (Proof.Events155.exact39710RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39709.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39709.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority39709.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39709.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority39709.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound39718

namespace LeftBound39719
def owner : Owner := ⟨.program ⟨214⟩, ⟨21843⟩⟩
def transferEvent : Nat := 39719
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 39718) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 39718)
      LeftBound39718.bound (LeftBound39718.actual selector witness) := by
  exact .transfer (LeftBound39718.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound39718.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound39718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound39718.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39719

namespace LeftBound39814
def owner : Owner := ⟨.program ⟨214⟩, ⟨16271⟩⟩
def transferEvent : Nat := 39814
def frameStart : Nat := 39775
def rule : BoundRule := .identity (.predecessor 0 39813 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39813 .coefficient)
      LeftAuthority39811.bound (LeftAuthority39811.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39812RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39811.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39811.derived selector witness)

def rawBound : CoeffClass := LeftAuthority39811.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39811.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority39811.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound39814

namespace LeftBound39831
def owner : Owner := ⟨.program ⟨214⟩, ⟨16345⟩⟩
def transferEvent : Nat := 39831
def frameStart : Nat := 39775
def rule : BoundRule := .sum [.predecessor 0 39829 .coefficient, .predecessor 1 39830 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39829 .coefficient)
      LeftBound39814.bound (LeftBound39814.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound39814.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39830 .coefficient)
      LeftAuthority39827.bound (LeftAuthority39827.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority39827.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39814.bound, LeftAuthority39827.bound]
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39814.bound, LeftAuthority39827.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39814.actual selector witness, LeftAuthority39827.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39831

namespace LeftBound39834
def owner : Owner := ⟨.program ⟨214⟩, ⟨16346⟩⟩
def transferEvent : Nat := 39834
def frameStart : Nat := 39775
def rule : BoundRule := .identity (.predecessor 0 39833 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39833 .coefficient)
      LeftBound39831.bound (LeftBound39831.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound39831.derived selector witness)

def rawBound : CoeffClass := LeftBound39831.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39831.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound39831.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound39834

namespace LeftBound39840
def owner : Owner := ⟨.program ⟨214⟩, ⟨16347⟩⟩
def transferEvent : Nat := 39840
def frameStart : Nat := 39775
def rule : BoundRule := .product (.predecessor 0 39838 .coefficient) (.predecessor 1 39839 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39838 .coefficient)
      LeftAuthority39836.bound (LeftAuthority39836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39837RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39836.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39836.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39839 .coefficient)
      LeftBound39834.bound (LeftBound39834.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39835RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39834.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39834.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority39836.bound LeftBound39834.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39836.bound, LeftBound39834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority39836.actual selector witness) * (LeftBound39834.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39840

namespace LeftBound39848
def owner : Owner := ⟨.program ⟨214⟩, ⟨16348⟩⟩
def transferEvent : Nat := 39848
def frameStart : Nat := 39775
def rule : BoundRule := .sum [.predecessor 0 39846 .coefficient, .predecessor 1 39847 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39846 .coefficient)
      LeftAuthority39844.bound (LeftAuthority39844.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39844.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39844.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39847 .coefficient)
      LeftBound39840.bound (LeftBound39840.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events155.exact39842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39840.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39840.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority39844.bound, LeftBound39840.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39844.bound, LeftBound39840.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority39844.actual selector witness, LeftBound39840.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39848

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
