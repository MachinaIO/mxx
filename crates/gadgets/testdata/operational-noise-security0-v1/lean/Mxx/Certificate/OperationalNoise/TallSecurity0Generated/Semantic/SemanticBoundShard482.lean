import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard481

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound70819
def owner : Owner := ⟨.program ⟨214⟩, ⟨7850⟩⟩
def transferEvent : Nat := 70819
def frameStart : Nat := 70744
def rule : BoundRule := .scale (.predecessor 0 70817 .coefficient) (.value (.predecessor 1 70818 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70817 .coefficient)
      LeftAuthority70815.bound (LeftAuthority70815.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70816RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70815.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70815.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70818 .coefficient)
      LeftAuthority70806.bound (LeftAuthority70806.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority70806.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority70815.bound LeftAuthority70806.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70815.bound, LeftAuthority70806.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority70815.actual selector witness) * (LeftAuthority70806.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound70819

namespace LeftBound70822
def owner : Owner := ⟨.program ⟨214⟩, ⟨6758⟩⟩
def transferEvent : Nat := 70822
def frameStart : Nat := 70744
def rule : BoundRule := .identity (.predecessor 0 70821 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70821 .coefficient)
      LeftAuthority70809.bound (LeftAuthority70809.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70810RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70809.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70809.derived selector witness)

def rawBound : CoeffClass := LeftAuthority70809.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70809.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority70809.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound70822

namespace LeftBound70826
def owner : Owner := ⟨.program ⟨214⟩, ⟨7851⟩⟩
def transferEvent : Nat := 70826
def frameStart : Nat := 70744
def rule : BoundRule := .product (.predecessor 0 70824 .coefficient) (.predecessor 1 70825 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70824 .coefficient)
      LeftBound70822.bound (LeftBound70822.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70823RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70822.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70822.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70825 .coefficient)
      LeftBound70819.bound (LeftBound70819.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70819.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70819.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70822.bound LeftBound70819.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70822.bound, LeftBound70819.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70822.actual selector witness) * (LeftBound70819.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70826

namespace LeftBound70831
def owner : Owner := ⟨.program ⟨214⟩, ⟨14096⟩⟩
def transferEvent : Nat := 70831
def frameStart : Nat := 70744
def rule : BoundRule := .sum [.predecessor 0 70829 .coefficient, .predecessor 1 70830 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70829 .coefficient)
      LeftBound70826.bound (LeftBound70826.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70828RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70826.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70826.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70830 .coefficient)
      LeftBound70803.bound (LeftBound70803.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70805RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70803.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70803.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70826.bound, LeftBound70803.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70826.bound, LeftBound70803.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70826.actual selector witness, LeftBound70803.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70831

namespace LeftBound70835
def owner : Owner := ⟨.program ⟨214⟩, ⟨25987⟩⟩
def transferEvent : Nat := 70835
def frameStart : Nat := 70744
def rule : BoundRule := .product (.predecessor 0 70833 .coefficient) (.predecessor 1 70834 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70833 .coefficient)
      LeftBound70831.bound (LeftBound70831.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70832RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70831.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70831.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70834 .coefficient)
      LeftAuthority70788.bound (LeftAuthority70788.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70789RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70788.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70788.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70831.bound LeftAuthority70788.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70831.bound, LeftAuthority70788.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70831.actual selector witness) * (LeftAuthority70788.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70835

namespace LeftBound70846
def owner : Owner := ⟨.program ⟨214⟩, ⟨15819⟩⟩
def transferEvent : Nat := 70846
def frameStart : Nat := 70744
def rule : BoundRule := .product (.predecessor 0 70844 .coefficient) (.predecessor 1 70845 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70844 .coefficient)
      LeftAuthority70799.bound (LeftAuthority70799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70800RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70799.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70799.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70845 .coefficient)
      LeftAuthority70842.bound (LeftAuthority70842.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70843RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70842.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70842.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority70799.bound LeftAuthority70842.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70799.bound, LeftAuthority70842.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority70799.actual selector witness) * (LeftAuthority70842.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70846

namespace LeftBound70854
def owner : Owner := ⟨.program ⟨214⟩, ⟨15820⟩⟩
def transferEvent : Nat := 70854
def frameStart : Nat := 70744
def rule : BoundRule := .sum [.predecessor 0 70852 .coefficient, .predecessor 1 70853 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70852 .coefficient)
      LeftAuthority70850.bound (LeftAuthority70850.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70851RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70850.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70850.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70853 .coefficient)
      LeftBound70846.bound (LeftBound70846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70848RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70846.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70846.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority70850.bound, LeftBound70846.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70850.bound, LeftBound70846.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority70850.actual selector witness, LeftBound70846.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70854

namespace LeftBound70858
def owner : Owner := ⟨.program ⟨214⟩, ⟨25988⟩⟩
def transferEvent : Nat := 70858
def frameStart : Nat := 70744
def rule : BoundRule := .sum [.predecessor 0 70856 .coefficient, .predecessor 1 70857 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70856 .coefficient)
      LeftBound70854.bound (LeftBound70854.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70855RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70854.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70854.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70857 .coefficient)
      LeftBound70835.bound (LeftBound70835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70835.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70835.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70854.bound, LeftBound70835.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70854.bound, LeftBound70835.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70854.actual selector witness, LeftBound70835.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70858

namespace LeftBound70871
def owner : Owner := ⟨.program ⟨214⟩, ⟨25986⟩⟩
def transferEvent : Nat := 70871
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 70869 .coefficient, .predecessor 1 70870 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70869 .coefficient)
      LeftBound70692.bound (LeftBound70692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70692.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70870 .coefficient)
      LeftBound70675.bound (LeftBound70675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70682RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70675.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70675.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70692.bound, LeftBound70675.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70692.bound, LeftBound70675.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70692.actual selector witness, LeftBound70675.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70871

namespace LeftBound70874
def owner : Owner := ⟨.program ⟨214⟩, ⟨25986⟩⟩
def transferEvent : Nat := 70874
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 70868 .summary, .result 70682 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70868 .summary)
      LeftBound70694.bound (LeftBound70694.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19455⟩⟩) (rawTerms := some (Proof.Events276.exact70868RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70694.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70682 .summary)
      LeftBound70677.bound (LeftBound70677.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25985⟩⟩) (rawTerms := some (Proof.Events276.exact70682RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70677.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70694.bound, LeftBound70677.bound]
def bound : CoeffClass := .finite ⟨352054612209664, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70694.bound, LeftBound70677.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70694.actual selector witness, LeftBound70677.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70874

namespace LeftBound70878
def owner : Owner := ⟨.program ⟨214⟩, ⟨27638⟩⟩
def transferEvent : Nat := 70878
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 70876 .coefficient) (.predecessor 1 70877 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70876 .coefficient)
      LeftBound70871.bound (LeftBound70871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70871.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70877 .coefficient)
      LeftAuthority70597.bound (LeftAuthority70597.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70598RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70597.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70597.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70871.bound LeftAuthority70597.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70871.bound, LeftAuthority70597.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70871.actual selector witness) * (LeftAuthority70597.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70878

namespace LeftBound70879
def owner : Owner := ⟨.program ⟨214⟩, ⟨27638⟩⟩
def transferEvent : Nat := 70879
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27636⟩⟩]⟩ [⟨.result 70598 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70598 .coefficient)
      LeftAuthority70597.bound (LeftAuthority70597.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27636⟩⟩) (rawTerms := some (Proof.Events275.exact70598RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70597.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70597.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority70597.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70597.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority70597.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound70879

namespace LeftBound70880
def owner : Owner := ⟨.program ⟨214⟩, ⟨27638⟩⟩
def transferEvent : Nat := 70880
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 70875 .summary) (.transfer 70879) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70875 .summary)
      LeftBound70874.bound (LeftBound70874.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25986⟩⟩) (rawTerms := some (Proof.Events276.exact70875RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70874.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 70879)
      LeftBound70879.bound (LeftBound70879.actual selector witness) := by
  exact .transfer (LeftBound70879.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70874.bound LeftBound70879.bound
def bound : CoeffClass := .finite ⟨1292046059683262234624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70874.bound, LeftBound70879.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70874.actual selector witness) * (LeftBound70879.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70880

namespace LeftBound70891
def owner : Owner := ⟨.program ⟨214⟩, ⟨21254⟩⟩
def transferEvent : Nat := 70891
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 70889 .coefficient) (.value (.predecessor 1 70890 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70889 .coefficient)
      LeftAuthority70887.bound (LeftAuthority70887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70887.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70887.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70890 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority70887.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70887.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority70887.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound70891

namespace LeftBound70895
def owner : Owner := ⟨.program ⟨214⟩, ⟨21255⟩⟩
def transferEvent : Nat := 70895
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 70893 .coefficient) (.predecessor 1 70894 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70893 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70894 .coefficient)
      LeftBound70891.bound (LeftBound70891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70891.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70891.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound70891.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound70891.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound70891.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70895

namespace LeftBound70896
def owner : Owner := ⟨.program ⟨214⟩, ⟨21255⟩⟩
def transferEvent : Nat := 70896
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21252⟩⟩]⟩ [⟨.result 70888 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70888 .coefficient)
      LeftAuthority70887.bound (LeftAuthority70887.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21252⟩⟩) (rawTerms := some (Proof.Events276.exact70888RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70887.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70887.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority70887.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70887.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority70887.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound70896

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
