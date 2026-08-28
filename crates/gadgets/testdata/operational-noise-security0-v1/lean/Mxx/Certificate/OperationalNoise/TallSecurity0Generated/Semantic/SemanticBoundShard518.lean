import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard464
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard517

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound76681
def owner : Owner := ⟨.program ⟨214⟩, ⟨22047⟩⟩
def transferEvent : Nat := 76681
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 76680) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 76680)
      LeftBound76680.bound (LeftBound76680.actual selector witness) := by
  exact .transfer (LeftBound76680.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound76680.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound76680.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound76680.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76681

namespace LeftBound76776
def owner : Owner := ⟨.program ⟨214⟩, ⟨16462⟩⟩
def transferEvent : Nat := 76776
def frameStart : Nat := 76737
def rule : BoundRule := .identity (.predecessor 0 76775 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76775 .coefficient)
      LeftAuthority76773.bound (LeftAuthority76773.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76774RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76773.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76773.derived selector witness)

def rawBound : CoeffClass := LeftAuthority76773.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76773.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority76773.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound76776

namespace LeftBound76793
def owner : Owner := ⟨.program ⟨214⟩, ⟨16501⟩⟩
def transferEvent : Nat := 76793
def frameStart : Nat := 76737
def rule : BoundRule := .sum [.predecessor 0 76791 .coefficient, .predecessor 1 76792 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76791 .coefficient)
      LeftBound76776.bound (LeftBound76776.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound76776.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76792 .coefficient)
      LeftAuthority76789.bound (LeftAuthority76789.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority76789.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76776.bound, LeftAuthority76789.bound]
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76776.bound, LeftAuthority76789.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound76776.actual selector witness, LeftAuthority76789.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76793

namespace LeftBound76796
def owner : Owner := ⟨.program ⟨214⟩, ⟨16502⟩⟩
def transferEvent : Nat := 76796
def frameStart : Nat := 76737
def rule : BoundRule := .identity (.predecessor 0 76795 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76795 .coefficient)
      LeftBound76793.bound (LeftBound76793.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound76793.derived selector witness)

def rawBound : CoeffClass := LeftBound76793.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76793.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound76793.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound76796

namespace LeftBound76802
def owner : Owner := ⟨.program ⟨214⟩, ⟨16503⟩⟩
def transferEvent : Nat := 76802
def frameStart : Nat := 76737
def rule : BoundRule := .product (.predecessor 0 76800 .coefficient) (.predecessor 1 76801 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76800 .coefficient)
      LeftAuthority76798.bound (LeftAuthority76798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76798.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76798.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76801 .coefficient)
      LeftBound76796.bound (LeftBound76796.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76797RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76796.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76796.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority76798.bound LeftBound76796.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76798.bound, LeftBound76796.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority76798.actual selector witness) * (LeftBound76796.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76802

namespace LeftBound76810
def owner : Owner := ⟨.program ⟨214⟩, ⟨16504⟩⟩
def transferEvent : Nat := 76810
def frameStart : Nat := 76737
def rule : BoundRule := .sum [.predecessor 0 76808 .coefficient, .predecessor 1 76809 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76808 .coefficient)
      LeftAuthority76806.bound (LeftAuthority76806.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact76807RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76806.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76806.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76809 .coefficient)
      LeftBound76802.bound (LeftBound76802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact76804RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76802.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76802.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority76806.bound, LeftBound76802.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76806.bound, LeftBound76802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority76806.actual selector witness, LeftBound76802.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76810

namespace LeftBound76814
def owner : Owner := ⟨.program ⟨214⟩, ⟨28932⟩⟩
def transferEvent : Nat := 76814
def frameStart : Nat := 76737
def rule : BoundRule := .product (.predecessor 0 76812 .coefficient) (.predecessor 1 76813 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76812 .coefficient)
      LeftBound76810.bound (LeftBound76810.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact76811RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76810.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76810.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76813 .coefficient)
      LeftAuthority76787.bound (LeftAuthority76787.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76788RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76787.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76787.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound76810.bound LeftAuthority76787.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76810.bound, LeftAuthority76787.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound76810.actual selector witness) * (LeftAuthority76787.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76814

namespace LeftBound76825
def owner : Owner := ⟨.program ⟨214⟩, ⟨17548⟩⟩
def transferEvent : Nat := 76825
def frameStart : Nat := 76737
def rule : BoundRule := .product (.predecessor 0 76823 .coefficient) (.predecessor 1 76824 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76823 .coefficient)
      LeftAuthority76798.bound (LeftAuthority76798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76798.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76798.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76824 .coefficient)
      LeftAuthority76821.bound (LeftAuthority76821.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact76822RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76821.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76821.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority76798.bound LeftAuthority76821.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76798.bound, LeftAuthority76821.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority76798.actual selector witness) * (LeftAuthority76821.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76825

namespace LeftBound76833
def owner : Owner := ⟨.program ⟨214⟩, ⟨17549⟩⟩
def transferEvent : Nat := 76833
def frameStart : Nat := 76737
def rule : BoundRule := .sum [.predecessor 0 76831 .coefficient, .predecessor 1 76832 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76831 .coefficient)
      LeftAuthority76829.bound (LeftAuthority76829.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact76830RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76829.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76829.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76832 .coefficient)
      LeftBound76825.bound (LeftBound76825.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact76827RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76825.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76825.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority76829.bound, LeftBound76825.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76829.bound, LeftBound76825.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority76829.actual selector witness, LeftBound76825.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76833

namespace LeftBound76837
def owner : Owner := ⟨.program ⟨214⟩, ⟨28937⟩⟩
def transferEvent : Nat := 76837
def frameStart : Nat := 76737
def rule : BoundRule := .sum [.predecessor 0 76835 .coefficient, .predecessor 1 76836 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76835 .coefficient)
      LeftBound76833.bound (LeftBound76833.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact76834RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76833.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76833.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76836 .coefficient)
      LeftBound76814.bound (LeftBound76814.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact76819RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76814.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76814.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76833.bound, LeftBound76814.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76833.bound, LeftBound76814.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound76833.actual selector witness, LeftBound76814.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76837

namespace LeftBound76850
def owner : Owner := ⟨.program ⟨214⟩, ⟨28934⟩⟩
def transferEvent : Nat := 76850
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 76848 .coefficient, .predecessor 1 76849 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76848 .coefficient)
      LeftBound76679.bound (LeftBound76679.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact76847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76679.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76679.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76849 .coefficient)
      LeftBound76662.bound (LeftBound76662.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76669RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76662.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76662.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76679.bound, LeftBound76662.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76679.bound, LeftBound76662.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound76679.actual selector witness, LeftBound76662.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76850

namespace LeftBound76853
def owner : Owner := ⟨.program ⟨214⟩, ⟨28934⟩⟩
def transferEvent : Nat := 76853
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 76847 .summary, .result 76669 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76847 .summary)
      LeftBound76681.bound (LeftBound76681.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22047⟩⟩) (rawTerms := some (Proof.Events300.exact76847RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76681.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76669 .summary)
      LeftBound76664.bound (LeftBound76664.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28933⟩⟩) (rawTerms := some (Proof.Events299.exact76669RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76664.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76681.bound, LeftBound76664.bound]
def bound : CoeffClass := .finite ⟨1292315010834812776448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76681.bound, LeftBound76664.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound76681.actual selector witness, LeftBound76664.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76853

namespace LeftBound76857
def owner : Owner := ⟨.program ⟨214⟩, ⟨28935⟩⟩
def transferEvent : Nat := 76857
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 76855 .coefficient) (.predecessor 1 76856 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76855 .coefficient)
      LeftBound76850.bound (LeftBound76850.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact76854RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76850.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76850.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76856 .coefficient)
      LeftBound5618.bound (LeftBound5618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5618.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5618.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound76850.bound LeftBound5618.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76850.bound, LeftBound5618.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound76850.actual selector witness) * (LeftBound5618.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76857

namespace LeftBound76858
def owner : Owner := ⟨.program ⟨214⟩, ⟨28935⟩⟩
def transferEvent : Nat := 76858
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩ [⟨.result 5615 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5615 .coefficient)
      LeftAuthority5614.bound (LeftAuthority5614.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6669⟩⟩) (rawTerms := some (Proof.Events021.exact5615RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5614.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5614.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5614.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5614.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound76858

namespace LeftBound76859
def owner : Owner := ⟨.program ⟨214⟩, ⟨28935⟩⟩
def transferEvent : Nat := 76859
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 76854 .summary) (.transfer 76858) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76854 .summary)
      LeftBound76853.bound (LeftBound76853.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28934⟩⟩) (rawTerms := some (Proof.Events300.exact76854RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76853.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 76858)
      LeftBound76858.bound (LeftBound76858.actual selector witness) := by
  exact .transfer (LeftBound76858.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound76853.bound LeftBound76858.bound
def bound : CoeffClass := .finite ⟨4742816766803936246568583168, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76853.bound, LeftBound76858.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound76853.actual selector witness) * (LeftBound76858.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76859

namespace LeftBound76874
def owner : Owner := ⟨.program ⟨214⟩, ⟨28716⟩⟩
def transferEvent : Nat := 76874
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 76872 .coefficient) (.predecessor 1 76873 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76872 .coefficient)
      LeftBound68461.bound (LeftBound68461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68465RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68461.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68461.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76873 .coefficient)
      LeftAuthority76870.bound (LeftAuthority76870.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events300.exact76871RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76870.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76870.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound68461.bound LeftAuthority76870.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68461.bound, LeftAuthority76870.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound68461.actual selector witness) * (LeftAuthority76870.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76874

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
