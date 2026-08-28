import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard076
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard237
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard279

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound41645
def owner : Owner := ⟨.program ⟨214⟩, ⟨21267⟩⟩
def transferEvent : Nat := 41645
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 41643 .coefficient) (.predecessor 1 41644 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41643 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41644 .coefficient)
      LeftBound41641.bound (LeftBound41641.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41642RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41641.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41641.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound41641.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound41641.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound41641.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41645

namespace LeftBound41646
def owner : Owner := ⟨.program ⟨214⟩, ⟨21267⟩⟩
def transferEvent : Nat := 41646
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21264⟩⟩]⟩ [⟨.result 41638 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41638 .coefficient)
      LeftAuthority41637.bound (LeftAuthority41637.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21264⟩⟩) (rawTerms := some (Proof.Events162.exact41638RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41637.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41637.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority41637.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41637.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority41637.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound41646

namespace LeftBound41647
def owner : Owner := ⟨.program ⟨214⟩, ⟨21267⟩⟩
def transferEvent : Nat := 41647
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 41646) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 41646)
      LeftBound41646.bound (LeftBound41646.actual selector witness) := by
  exact .transfer (LeftBound41646.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound41646.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound41646.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound41646.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41647

namespace LeftBound41742
def owner : Owner := ⟨.program ⟨214⟩, ⟨15830⟩⟩
def transferEvent : Nat := 41742
def frameStart : Nat := 41703
def rule : BoundRule := .identity (.predecessor 0 41741 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41741 .coefficient)
      LeftAuthority41739.bound (LeftAuthority41739.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41739.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41739.derived selector witness)

def rawBound : CoeffClass := LeftAuthority41739.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41739.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority41739.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound41742

namespace LeftBound41759
def owner : Owner := ⟨.program ⟨214⟩, ⟨15904⟩⟩
def transferEvent : Nat := 41759
def frameStart : Nat := 41703
def rule : BoundRule := .sum [.predecessor 0 41757 .coefficient, .predecessor 1 41758 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41757 .coefficient)
      LeftBound41742.bound (LeftBound41742.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound41742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41758 .coefficient)
      LeftAuthority41755.bound (LeftAuthority41755.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority41755.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41742.bound, LeftAuthority41755.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41742.bound, LeftAuthority41755.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41742.actual selector witness, LeftAuthority41755.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41759

namespace LeftBound41762
def owner : Owner := ⟨.program ⟨214⟩, ⟨15905⟩⟩
def transferEvent : Nat := 41762
def frameStart : Nat := 41703
def rule : BoundRule := .identity (.predecessor 0 41761 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41761 .coefficient)
      LeftBound41759.bound (LeftBound41759.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound41759.derived selector witness)

def rawBound : CoeffClass := LeftBound41759.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41759.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound41759.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound41762

namespace LeftBound41768
def owner : Owner := ⟨.program ⟨214⟩, ⟨15906⟩⟩
def transferEvent : Nat := 41768
def frameStart : Nat := 41703
def rule : BoundRule := .product (.predecessor 0 41766 .coefficient) (.predecessor 1 41767 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41766 .coefficient)
      LeftAuthority41764.bound (LeftAuthority41764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41765RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41764.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41764.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41767 .coefficient)
      LeftBound41762.bound (LeftBound41762.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41763RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41762.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41762.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority41764.bound LeftBound41762.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41764.bound, LeftBound41762.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority41764.actual selector witness) * (LeftBound41762.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41768

namespace LeftBound41776
def owner : Owner := ⟨.program ⟨214⟩, ⟨15907⟩⟩
def transferEvent : Nat := 41776
def frameStart : Nat := 41703
def rule : BoundRule := .sum [.predecessor 0 41774 .coefficient, .predecessor 1 41775 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41774 .coefficient)
      LeftAuthority41772.bound (LeftAuthority41772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41773RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41772.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41772.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41775 .coefficient)
      LeftBound41768.bound (LeftBound41768.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41770RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41768.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41768.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority41772.bound, LeftBound41768.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41772.bound, LeftBound41768.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority41772.actual selector witness, LeftBound41768.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41776

namespace LeftBound41780
def owner : Owner := ⟨.program ⟨214⟩, ⟨27676⟩⟩
def transferEvent : Nat := 41780
def frameStart : Nat := 41703
def rule : BoundRule := .product (.predecessor 0 41778 .coefficient) (.predecessor 1 41779 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41778 .coefficient)
      LeftBound41776.bound (LeftBound41776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41777RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41776.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41779 .coefficient)
      LeftAuthority41753.bound (LeftAuthority41753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41754RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41753.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41753.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41776.bound LeftAuthority41753.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41776.bound, LeftAuthority41753.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41776.actual selector witness) * (LeftAuthority41753.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41780

namespace LeftBound41791
def owner : Owner := ⟨.program ⟨214⟩, ⟨15874⟩⟩
def transferEvent : Nat := 41791
def frameStart : Nat := 41703
def rule : BoundRule := .product (.predecessor 0 41789 .coefficient) (.predecessor 1 41790 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41789 .coefficient)
      LeftAuthority41764.bound (LeftAuthority41764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41765RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41764.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41764.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41790 .coefficient)
      LeftAuthority41787.bound (LeftAuthority41787.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41788RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41787.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41787.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority41764.bound LeftAuthority41787.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41764.bound, LeftAuthority41787.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority41764.actual selector witness) * (LeftAuthority41787.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41791

namespace LeftBound41799
def owner : Owner := ⟨.program ⟨214⟩, ⟨15875⟩⟩
def transferEvent : Nat := 41799
def frameStart : Nat := 41703
def rule : BoundRule := .sum [.predecessor 0 41797 .coefficient, .predecessor 1 41798 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41797 .coefficient)
      LeftAuthority41795.bound (LeftAuthority41795.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41796RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41795.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41795.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41798 .coefficient)
      LeftBound41791.bound (LeftBound41791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41793RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41791.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41791.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority41795.bound, LeftBound41791.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41795.bound, LeftBound41791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority41795.actual selector witness, LeftBound41791.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41799

namespace LeftBound41803
def owner : Owner := ⟨.program ⟨214⟩, ⟨27680⟩⟩
def transferEvent : Nat := 41803
def frameStart : Nat := 41703
def rule : BoundRule := .sum [.predecessor 0 41801 .coefficient, .predecessor 1 41802 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41801 .coefficient)
      LeftBound41799.bound (LeftBound41799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41800RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41799.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41799.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41802 .coefficient)
      LeftBound41780.bound (LeftBound41780.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41785RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41780.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41780.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41799.bound, LeftBound41780.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41799.bound, LeftBound41780.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41799.actual selector witness, LeftBound41780.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41803

namespace LeftBound41816
def owner : Owner := ⟨.program ⟨214⟩, ⟨27678⟩⟩
def transferEvent : Nat := 41816
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 41814 .coefficient, .predecessor 1 41815 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41814 .coefficient)
      LeftBound41645.bound (LeftBound41645.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41813RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41645.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41645.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41815 .coefficient)
      LeftBound41628.bound (LeftBound41628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41628.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41628.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41645.bound, LeftBound41628.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41645.bound, LeftBound41628.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41645.actual selector witness, LeftBound41628.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41816

namespace LeftBound41819
def owner : Owner := ⟨.program ⟨214⟩, ⟨27678⟩⟩
def transferEvent : Nat := 41819
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 41813 .summary, .result 41635 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41813 .summary)
      LeftBound41647.bound (LeftBound41647.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21267⟩⟩) (rawTerms := some (Proof.Events163.exact41813RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound41647.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41635 .summary)
      LeftBound41630.bound (LeftBound41630.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27677⟩⟩) (rawTerms := some (Proof.Events162.exact41635RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound41630.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41647.bound, LeftBound41630.bound]
def bound : CoeffClass := .finite ⟨1292046061494565744640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41647.bound, LeftBound41630.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41647.actual selector witness, LeftBound41630.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41819

namespace LeftBound41843
def owner : Owner := ⟨.program ⟨214⟩, ⟨11310⟩⟩
def transferEvent : Nat := 41843
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 41841 .coefficient) (.predecessor 1 41842 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41841 .coefficient)
      LeftAuthority1865.bound (LeftAuthority1865.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events007.exact1866RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1865.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1865.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41842 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1865.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1865.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1865.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound41843

namespace LeftBound41848
def owner : Owner := ⟨.program ⟨214⟩, ⟨7309⟩⟩
def transferEvent : Nat := 41848
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 41846 .coefficient) (.predecessor 1 41847 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41846 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41847 .coefficient)
      LeftBound12483.bound (LeftBound12483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12483.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12483.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound12483.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound12483.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound12483.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41848

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
