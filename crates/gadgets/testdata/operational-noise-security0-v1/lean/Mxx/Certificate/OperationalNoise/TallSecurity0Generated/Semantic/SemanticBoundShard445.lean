import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard444

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound65855
def owner : Owner := ⟨.program ⟨214⟩, ⟨25677⟩⟩
def transferEvent : Nat := 65855
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 65853 .coefficient) (.predecessor 1 65854 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65853 .coefficient)
      LeftBound65849.bound (LeftBound65849.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact65852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65849.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65849.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65854 .coefficient)
      LeftAuthority65787.bound (LeftAuthority65787.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65788RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65787.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65787.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound65849.bound LeftAuthority65787.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65849.bound, LeftAuthority65787.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound65849.actual selector witness) * (LeftAuthority65787.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65855

namespace LeftBound65856
def owner : Owner := ⟨.program ⟨214⟩, ⟨25677⟩⟩
def transferEvent : Nat := 65856
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25676⟩⟩]⟩ [⟨.result 65788 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65788 .coefficient)
      LeftAuthority65787.bound (LeftAuthority65787.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25676⟩⟩) (rawTerms := some (Proof.Events256.exact65788RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65787.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65787.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority65787.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65787.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority65787.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound65856

namespace LeftBound65857
def owner : Owner := ⟨.program ⟨214⟩, ⟨25677⟩⟩
def transferEvent : Nat := 65857
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65852 .summary) (.transfer 65856) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65852 .summary)
      LeftBound65851.bound (LeftBound65851.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13153⟩⟩) (rawTerms := some (Proof.Events257.exact65852RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65851.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65856)
      LeftBound65856.bound (LeftBound65856.actual selector witness) := by
  exact .transfer (LeftBound65856.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound65851.bound LeftBound65856.bound
def bound : CoeffClass := .finite ⟨350371553738752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65851.bound, LeftBound65856.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound65851.actual selector witness) * (LeftBound65856.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65857

namespace LeftBound65868
def owner : Owner := ⟨.program ⟨214⟩, ⟨20174⟩⟩
def transferEvent : Nat := 65868
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 65866 .coefficient) (.value (.predecessor 1 65867 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65866 .coefficient)
      LeftAuthority65864.bound (LeftAuthority65864.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact65865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65864.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65864.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65867 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority65864.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65864.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority65864.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound65868

namespace LeftBound65872
def owner : Owner := ⟨.program ⟨214⟩, ⟨20175⟩⟩
def transferEvent : Nat := 65872
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 65870 .coefficient) (.predecessor 1 65871 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65870 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65871 .coefficient)
      LeftBound65868.bound (LeftBound65868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact65869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65868.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65868.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound65868.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound65868.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound65868.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65872

namespace LeftBound65873
def owner : Owner := ⟨.program ⟨214⟩, ⟨20175⟩⟩
def transferEvent : Nat := 65873
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20172⟩⟩]⟩ [⟨.result 65865 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65865 .coefficient)
      LeftAuthority65864.bound (LeftAuthority65864.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20172⟩⟩) (rawTerms := some (Proof.Events257.exact65865RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65864.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65864.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority65864.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65864.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority65864.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound65873

namespace LeftBound65874
def owner : Owner := ⟨.program ⟨214⟩, ⟨20175⟩⟩
def transferEvent : Nat := 65874
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 65873) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65873)
      LeftBound65873.bound (LeftBound65873.actual selector witness) := by
  exact .transfer (LeftBound65873.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound65873.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound65873.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound65873.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65874

namespace LeftBound65953
def owner : Owner := ⟨.program ⟨214⟩, ⟨13147⟩⟩
def transferEvent : Nat := 65953
def frameStart : Nat := 65924
def rule : BoundRule := .product (.predecessor 0 65951 .coefficient) (.predecessor 1 65952 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65951 .coefficient)
      LeftAuthority65949.bound (LeftAuthority65949.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact65950RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65949.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65949.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65952 .coefficient)
      LeftAuthority65946.bound (LeftAuthority65946.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact65947RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65946.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65946.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority65949.bound LeftAuthority65946.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65949.bound, LeftAuthority65946.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority65949.actual selector witness) * (LeftAuthority65946.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65953

namespace LeftBound65957
def owner : Owner := ⟨.program ⟨214⟩, ⟨13148⟩⟩
def transferEvent : Nat := 65957
def frameStart : Nat := 65924
def rule : BoundRule := .identity (.predecessor 0 65956 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65956 .coefficient)
      LeftBound65953.bound (LeftBound65953.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact65955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65953.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65953.derived selector witness)

def rawBound : CoeffClass := LeftBound65953.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65953.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound65953.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound65957

namespace LeftBound65974
def owner : Owner := ⟨.program ⟨214⟩, ⟨13246⟩⟩
def transferEvent : Nat := 65974
def frameStart : Nat := 65924
def rule : BoundRule := .sum [.predecessor 0 65972 .coefficient, .predecessor 1 65973 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65972 .coefficient)
      LeftBound65957.bound (LeftBound65957.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound65957.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65973 .coefficient)
      LeftAuthority65970.bound (LeftAuthority65970.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority65970.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65957.bound, LeftAuthority65970.bound]
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65957.bound, LeftAuthority65970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65957.actual selector witness, LeftAuthority65970.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65974

namespace LeftBound65977
def owner : Owner := ⟨.program ⟨214⟩, ⟨13247⟩⟩
def transferEvent : Nat := 65977
def frameStart : Nat := 65924
def rule : BoundRule := .identity (.predecessor 0 65976 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65976 .coefficient)
      LeftBound65974.bound (LeftBound65974.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound65974.derived selector witness)

def rawBound : CoeffClass := LeftBound65974.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound65974.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound65977

namespace LeftBound65983
def owner : Owner := ⟨.program ⟨214⟩, ⟨13248⟩⟩
def transferEvent : Nat := 65983
def frameStart : Nat := 65924
def rule : BoundRule := .product (.predecessor 0 65981 .coefficient) (.predecessor 1 65982 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65981 .coefficient)
      LeftAuthority65979.bound (LeftAuthority65979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact65980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65979.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65982 .coefficient)
      LeftBound65977.bound (LeftBound65977.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact65978RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65977.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65977.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority65979.bound LeftBound65977.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65979.bound, LeftBound65977.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority65979.actual selector witness) * (LeftBound65977.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65983

namespace LeftBound65999
def owner : Owner := ⟨.program ⟨214⟩, ⟨7880⟩⟩
def transferEvent : Nat := 65999
def frameStart : Nat := 65924
def rule : BoundRule := .scale (.predecessor 0 65997 .coefficient) (.value (.predecessor 1 65998 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65997 .coefficient)
      LeftAuthority65995.bound (LeftAuthority65995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact65996RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65995.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65995.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65998 .coefficient)
      LeftAuthority65986.bound (LeftAuthority65986.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority65986.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority65995.bound LeftAuthority65986.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65995.bound, LeftAuthority65986.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority65995.actual selector witness) * (LeftAuthority65986.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound65999

namespace LeftBound66002
def owner : Owner := ⟨.program ⟨214⟩, ⟨6769⟩⟩
def transferEvent : Nat := 66002
def frameStart : Nat := 65924
def rule : BoundRule := .identity (.predecessor 0 66001 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66001 .coefficient)
      LeftAuthority65989.bound (LeftAuthority65989.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact65990RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65989.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65989.derived selector witness)

def rawBound : CoeffClass := LeftAuthority65989.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65989.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority65989.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound66002

namespace LeftBound66006
def owner : Owner := ⟨.program ⟨214⟩, ⟨7881⟩⟩
def transferEvent : Nat := 66006
def frameStart : Nat := 65924
def rule : BoundRule := .product (.predecessor 0 66004 .coefficient) (.predecessor 1 66005 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66004 .coefficient)
      LeftBound66002.bound (LeftBound66002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact66003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66002.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66002.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66005 .coefficient)
      LeftBound65999.bound (LeftBound65999.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact66000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65999.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65999.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66002.bound LeftBound65999.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66002.bound, LeftBound65999.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66002.actual selector witness) * (LeftBound65999.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66006

namespace LeftBound66011
def owner : Owner := ⟨.program ⟨214⟩, ⟨13249⟩⟩
def transferEvent : Nat := 66011
def frameStart : Nat := 65924
def rule : BoundRule := .sum [.predecessor 0 66009 .coefficient, .predecessor 1 66010 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66009 .coefficient)
      LeftBound66006.bound (LeftBound66006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact66008RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66006.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66006.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66010 .coefficient)
      LeftBound65983.bound (LeftBound65983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact65985RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65983.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65983.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66006.bound, LeftBound65983.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66006.bound, LeftBound65983.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66006.actual selector witness, LeftBound65983.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66011

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
