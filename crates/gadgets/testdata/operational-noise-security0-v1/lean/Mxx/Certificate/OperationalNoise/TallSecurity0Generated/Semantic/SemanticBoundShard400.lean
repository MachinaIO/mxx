import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard095
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard399

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound58811
def owner : Owner := ⟨.program ⟨214⟩, ⟨15000⟩⟩
def transferEvent : Nat := 58811
def frameStart : Nat := 58738
def rule : BoundRule := .sum [.predecessor 0 58809 .coefficient, .predecessor 1 58810 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58809 .coefficient)
      LeftAuthority58807.bound (LeftAuthority58807.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58808RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58807.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58807.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58810 .coefficient)
      LeftBound58803.bound (LeftBound58803.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58805RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58803.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58803.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority58807.bound, LeftBound58803.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58807.bound, LeftBound58803.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority58807.actual selector witness, LeftBound58803.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58811

namespace LeftBound58815
def owner : Owner := ⟨.program ⟨214⟩, ⟨26578⟩⟩
def transferEvent : Nat := 58815
def frameStart : Nat := 58738
def rule : BoundRule := .product (.predecessor 0 58813 .coefficient) (.predecessor 1 58814 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58813 .coefficient)
      LeftBound58811.bound (LeftBound58811.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58812RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58811.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58811.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58814 .coefficient)
      LeftAuthority58788.bound (LeftAuthority58788.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58789RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58788.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58788.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound58811.bound LeftAuthority58788.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58811.bound, LeftAuthority58788.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound58811.actual selector witness) * (LeftAuthority58788.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58815

namespace LeftBound58826
def owner : Owner := ⟨.program ⟨214⟩, ⟨15316⟩⟩
def transferEvent : Nat := 58826
def frameStart : Nat := 58738
def rule : BoundRule := .product (.predecessor 0 58824 .coefficient) (.predecessor 1 58825 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58824 .coefficient)
      LeftAuthority58799.bound (LeftAuthority58799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58800RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58799.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58799.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58825 .coefficient)
      LeftAuthority58822.bound (LeftAuthority58822.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58823RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58822.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58822.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority58799.bound LeftAuthority58822.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58799.bound, LeftAuthority58822.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority58799.actual selector witness) * (LeftAuthority58822.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58826

namespace LeftBound58834
def owner : Owner := ⟨.program ⟨214⟩, ⟨15317⟩⟩
def transferEvent : Nat := 58834
def frameStart : Nat := 58738
def rule : BoundRule := .sum [.predecessor 0 58832 .coefficient, .predecessor 1 58833 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58832 .coefficient)
      LeftAuthority58830.bound (LeftAuthority58830.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58831RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58830.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58830.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58833 .coefficient)
      LeftBound58826.bound (LeftBound58826.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58828RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58826.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58826.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority58830.bound, LeftBound58826.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58830.bound, LeftBound58826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority58830.actual selector witness, LeftBound58826.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58834

namespace LeftBound58838
def owner : Owner := ⟨.program ⟨214⟩, ⟨26582⟩⟩
def transferEvent : Nat := 58838
def frameStart : Nat := 58738
def rule : BoundRule := .sum [.predecessor 0 58836 .coefficient, .predecessor 1 58837 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58836 .coefficient)
      LeftBound58834.bound (LeftBound58834.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58835RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58834.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58834.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58837 .coefficient)
      LeftBound58815.bound (LeftBound58815.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58815.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58815.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58834.bound, LeftBound58815.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58834.bound, LeftBound58815.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58834.actual selector witness, LeftBound58815.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58838

namespace LeftBound58851
def owner : Owner := ⟨.program ⟨214⟩, ⟨26580⟩⟩
def transferEvent : Nat := 58851
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 58849 .coefficient, .predecessor 1 58850 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58849 .coefficient)
      LeftBound58680.bound (LeftBound58680.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58848RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58680.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58680.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58850 .coefficient)
      LeftBound58663.bound (LeftBound58663.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58663.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58663.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58680.bound, LeftBound58663.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58680.bound, LeftBound58663.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58680.actual selector witness, LeftBound58663.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58851

namespace LeftBound58854
def owner : Owner := ⟨.program ⟨214⟩, ⟨26580⟩⟩
def transferEvent : Nat := 58854
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 58848 .summary, .result 58670 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58848 .summary)
      LeftBound58682.bound (LeftBound58682.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20543⟩⟩) (rawTerms := some (Proof.Events229.exact58848RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58682.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58670 .summary)
      LeftBound58665.bound (LeftBound58665.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26579⟩⟩) (rawTerms := some (Proof.Events229.exact58670RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58665.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58682.bound, LeftBound58665.bound]
def bound : CoeffClass := .finite ⟨1291900380601931935744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58682.bound, LeftBound58665.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58682.actual selector witness, LeftBound58665.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58854

namespace LeftBound58878
def owner : Owner := ⟨.program ⟨214⟩, ⟨10491⟩⟩
def transferEvent : Nat := 58878
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 58876 .coefficient) (.predecessor 1 58877 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58876 .coefficient)
      LeftAuthority2728.bound (LeftAuthority2728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events010.exact2729RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2728.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58877 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2728.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2728.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2728.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound58878

namespace LeftBound58883
def owner : Owner := ⟨.program ⟨214⟩, ⟨7266⟩⟩
def transferEvent : Nat := 58883
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 58881 .coefficient) (.predecessor 1 58882 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58881 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58882 .coefficient)
      LeftBound14988.bound (LeftBound14988.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14989RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14988.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14988.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound14988.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound14988.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound14988.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58883

namespace LeftBound58888
def owner : Owner := ⟨.program ⟨214⟩, ⟨10492⟩⟩
def transferEvent : Nat := 58888
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 58886 .coefficient, .predecessor 1 58887 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58886 .coefficient)
      LeftBound58883.bound (LeftBound58883.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact58885RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58883.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58883.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58887 .coefficient)
      LeftBound58878.bound (LeftBound58878.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact58880RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58878.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58878.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58883.bound, LeftBound58878.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58883.bound, LeftBound58878.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58883.actual selector witness, LeftBound58878.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58888

namespace LeftBound58892
def owner : Owner := ⟨.program ⟨214⟩, ⟨10493⟩⟩
def transferEvent : Nat := 58892
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 58890 .coefficient, .predecessor 1 58891 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58890 .coefficient)
      LeftBound58888.bound (LeftBound58888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact58889RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58888.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58888.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58891 .coefficient)
      LeftBound14980.bound (LeftBound14980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14980.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58888.bound, LeftBound14980.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58888.bound, LeftBound14980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58888.actual selector witness, LeftBound14980.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58892

namespace LeftBound58893
def owner : Owner := ⟨.program ⟨214⟩, ⟨10493⟩⟩
def transferEvent : Nat := 58893
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨86⟩⟩]⟩ [⟨.result 14981 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14981 .coefficient)
      LeftBound14980.bound (LeftBound14980.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨86⟩⟩) (rawTerms := some (Proof.Events058.exact14981RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14980.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound14980.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound14980.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound58893

namespace LeftBound58898
def owner : Owner := ⟨.program ⟨214⟩, ⟨10494⟩⟩
def transferEvent : Nat := 58898
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 58896 .coefficient) (.predecessor 1 58897 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58896 .coefficient)
      LeftBound58892.bound (LeftBound58892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events230.exact58895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58892.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58892.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58897 .coefficient)
      LeftAuthority2731.bound (LeftAuthority2731.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events010.exact2732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2731.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2731.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound58892.bound LeftAuthority2731.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58892.bound, LeftAuthority2731.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound58892.actual selector witness) * (LeftAuthority2731.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58898

namespace LeftBound58899
def owner : Owner := ⟨.program ⟨214⟩, ⟨10494⟩⟩
def transferEvent : Nat := 58899
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9405⟩⟩], []⟩ [⟨.result 2732 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2732 .coefficient)
      LeftAuthority2731.bound (LeftAuthority2731.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9405⟩⟩) (rawTerms := some (Proof.Events010.exact2732RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2731.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2731.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority2731.bound []
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2731.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority2731.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound58899

namespace LeftBound58900
def owner : Owner := ⟨.program ⟨214⟩, ⟨10494⟩⟩
def transferEvent : Nat := 58900
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 58895 .summary) (.transfer 58899) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58895 .summary)
      LeftBound58893.bound (LeftBound58893.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10493⟩⟩) (rawTerms := some (Proof.Events230.exact58895RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58893.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 58899)
      LeftBound58899.bound (LeftBound58899.actual selector witness) := by
  exact .transfer (LeftBound58899.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound58893.bound LeftBound58899.bound
def bound : CoeffClass := .finite ⟨1664, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58893.bound, LeftBound58899.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound58893.actual selector witness) * (LeftBound58899.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58900

namespace LeftBound58906
def owner : Owner := ⟨.program ⟨214⟩, ⟨9406⟩⟩
def transferEvent : Nat := 58906
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 58904 .coefficient) (.predecessor 1 58905 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58904 .coefficient)
      LeftAuthority2731.bound (LeftAuthority2731.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events010.exact2732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2731.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2731.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58905 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2731.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2731.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2731.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound58906

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
