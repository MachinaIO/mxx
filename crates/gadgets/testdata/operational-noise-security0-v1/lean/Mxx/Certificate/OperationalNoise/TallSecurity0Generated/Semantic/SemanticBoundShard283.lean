import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard282

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound42067
def owner : Owner := ⟨.program ⟨214⟩, ⟨25925⟩⟩
def transferEvent : Nat := 42067
def frameStart : Nat := 41976
def rule : BoundRule := .product (.predecessor 0 42065 .coefficient) (.predecessor 1 42066 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42065 .coefficient)
      LeftBound42063.bound (LeftBound42063.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42063.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42063.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42066 .coefficient)
      LeftAuthority42020.bound (LeftAuthority42020.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42020.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42020.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound42063.bound LeftAuthority42020.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42063.bound, LeftAuthority42020.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound42063.actual selector witness) * (LeftAuthority42020.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42067

namespace LeftBound42078
def owner : Owner := ⟨.program ⟨214⟩, ⟨15712⟩⟩
def transferEvent : Nat := 42078
def frameStart : Nat := 41976
def rule : BoundRule := .product (.predecessor 0 42076 .coefficient) (.predecessor 1 42077 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42076 .coefficient)
      LeftAuthority42031.bound (LeftAuthority42031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42031.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42031.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42077 .coefficient)
      LeftAuthority42074.bound (LeftAuthority42074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42074.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42074.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority42031.bound LeftAuthority42074.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42031.bound, LeftAuthority42074.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority42031.actual selector witness) * (LeftAuthority42074.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42078

namespace LeftBound42086
def owner : Owner := ⟨.program ⟨214⟩, ⟨15713⟩⟩
def transferEvent : Nat := 42086
def frameStart : Nat := 41976
def rule : BoundRule := .sum [.predecessor 0 42084 .coefficient, .predecessor 1 42085 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42084 .coefficient)
      LeftAuthority42082.bound (LeftAuthority42082.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42083RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42082.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42082.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42085 .coefficient)
      LeftBound42078.bound (LeftBound42078.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42078.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42078.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority42082.bound, LeftBound42078.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42082.bound, LeftBound42078.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority42082.actual selector witness, LeftBound42078.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42086

namespace LeftBound42090
def owner : Owner := ⟨.program ⟨214⟩, ⟨25926⟩⟩
def transferEvent : Nat := 42090
def frameStart : Nat := 41976
def rule : BoundRule := .sum [.predecessor 0 42088 .coefficient, .predecessor 1 42089 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42088 .coefficient)
      LeftBound42086.bound (LeftBound42086.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42087RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42086.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42086.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42089 .coefficient)
      LeftBound42067.bound (LeftBound42067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42067.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42086.bound, LeftBound42067.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42086.bound, LeftBound42067.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42086.actual selector witness, LeftBound42067.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42090

namespace LeftBound42103
def owner : Owner := ⟨.program ⟨214⟩, ⟨25924⟩⟩
def transferEvent : Nat := 42103
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 42101 .coefficient, .predecessor 1 42102 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42101 .coefficient)
      LeftBound41924.bound (LeftBound41924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42100RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41924.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41924.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42102 .coefficient)
      LeftBound41907.bound (LeftBound41907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41907.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41924.bound, LeftBound41907.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41924.bound, LeftBound41907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41924.actual selector witness, LeftBound41907.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42103

namespace LeftBound42106
def owner : Owner := ⟨.program ⟨214⟩, ⟨25924⟩⟩
def transferEvent : Nat := 42106
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 42100 .summary, .result 41914 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42100 .summary)
      LeftBound41926.bound (LeftBound41926.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19395⟩⟩) (rawTerms := some (Proof.Events164.exact42100RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound41926.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41914 .summary)
      LeftBound41909.bound (LeftBound41909.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25923⟩⟩) (rawTerms := some (Proof.Events163.exact41914RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound41909.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41926.bound, LeftBound41909.bound]
def bound : CoeffClass := .finite ⟨352042398396416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41926.bound, LeftBound41909.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41926.actual selector witness, LeftBound41909.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42106

namespace LeftBound42110
def owner : Owner := ⟨.program ⟨214⟩, ⟨27460⟩⟩
def transferEvent : Nat := 42110
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 42108 .coefficient) (.predecessor 1 42109 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42108 .coefficient)
      LeftBound42103.bound (LeftBound42103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42107RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42103.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42103.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42109 .coefficient)
      LeftAuthority41829.bound (LeftAuthority41829.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41830RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41829.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41829.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound42103.bound LeftAuthority41829.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42103.bound, LeftAuthority41829.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound42103.actual selector witness) * (LeftAuthority41829.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42110

namespace LeftBound42111
def owner : Owner := ⟨.program ⟨214⟩, ⟨27460⟩⟩
def transferEvent : Nat := 42111
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27458⟩⟩]⟩ [⟨.result 41830 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41830 .coefficient)
      LeftAuthority41829.bound (LeftAuthority41829.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27458⟩⟩) (rawTerms := some (Proof.Events163.exact41830RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41829.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41829.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority41829.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41829.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority41829.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound42111

namespace LeftBound42112
def owner : Owner := ⟨.program ⟨214⟩, ⟨27460⟩⟩
def transferEvent : Nat := 42112
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 42107 .summary) (.transfer 42111) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42107 .summary)
      LeftBound42106.bound (LeftBound42106.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25924⟩⟩) (rawTerms := some (Proof.Events164.exact42107RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound42106.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 42111)
      LeftBound42111.bound (LeftBound42111.actual selector witness) := by
  exact .transfer (LeftBound42111.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound42106.bound LeftBound42111.bound
def bound : CoeffClass := .finite ⟨1292001234793221062656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42106.bound, LeftBound42111.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound42106.actual selector witness) * (LeftBound42111.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42112

namespace LeftBound42123
def owner : Owner := ⟨.program ⟨214⟩, ⟨21122⟩⟩
def transferEvent : Nat := 42123
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 42121 .coefficient) (.value (.predecessor 1 42122 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42121 .coefficient)
      LeftAuthority42119.bound (LeftAuthority42119.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42119.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42119.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42122 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority42119.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42119.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority42119.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound42123

namespace LeftBound42127
def owner : Owner := ⟨.program ⟨214⟩, ⟨21123⟩⟩
def transferEvent : Nat := 42127
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 42125 .coefficient) (.predecessor 1 42126 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42125 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42126 .coefficient)
      LeftBound42123.bound (LeftBound42123.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42124RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42123.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42123.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound42123.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound42123.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound42123.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42127

namespace LeftBound42128
def owner : Owner := ⟨.program ⟨214⟩, ⟨21123⟩⟩
def transferEvent : Nat := 42128
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21120⟩⟩]⟩ [⟨.result 42120 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42120 .coefficient)
      LeftAuthority42119.bound (LeftAuthority42119.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21120⟩⟩) (rawTerms := some (Proof.Events164.exact42120RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42119.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42119.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority42119.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42119.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority42119.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound42128

namespace LeftBound42129
def owner : Owner := ⟨.program ⟨214⟩, ⟨21123⟩⟩
def transferEvent : Nat := 42129
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 42128) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 42128)
      LeftBound42128.bound (LeftBound42128.actual selector witness) := by
  exact .transfer (LeftBound42128.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound42128.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound42128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound42128.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42129

namespace LeftBound42224
def owner : Owner := ⟨.program ⟨214⟩, ⟨15711⟩⟩
def transferEvent : Nat := 42224
def frameStart : Nat := 42185
def rule : BoundRule := .identity (.predecessor 0 42223 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42223 .coefficient)
      LeftAuthority42221.bound (LeftAuthority42221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42222RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42221.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42221.derived selector witness)

def rawBound : CoeffClass := LeftAuthority42221.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42221.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority42221.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound42224

namespace LeftBound42241
def owner : Owner := ⟨.program ⟨214⟩, ⟨15785⟩⟩
def transferEvent : Nat := 42241
def frameStart : Nat := 42185
def rule : BoundRule := .sum [.predecessor 0 42239 .coefficient, .predecessor 1 42240 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42239 .coefficient)
      LeftBound42224.bound (LeftBound42224.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound42224.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42240 .coefficient)
      LeftAuthority42237.bound (LeftAuthority42237.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority42237.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42224.bound, LeftAuthority42237.bound]
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42224.bound, LeftAuthority42237.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42224.actual selector witness, LeftAuthority42237.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42241

namespace LeftBound42244
def owner : Owner := ⟨.program ⟨214⟩, ⟨15786⟩⟩
def transferEvent : Nat := 42244
def frameStart : Nat := 42185
def rule : BoundRule := .identity (.predecessor 0 42243 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42243 .coefficient)
      LeftBound42241.bound (LeftBound42241.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound42241.derived selector witness)

def rawBound : CoeffClass := LeftBound42241.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42241.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound42241.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound42244

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
