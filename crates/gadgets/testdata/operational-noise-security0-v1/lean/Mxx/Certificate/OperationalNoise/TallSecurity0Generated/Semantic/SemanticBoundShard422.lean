import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard377
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard421

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound63019
def owner : Owner := ⟨.program ⟨214⟩, ⟨16139⟩⟩
def transferEvent : Nat := 63019
def frameStart : Nat := 62960
def rule : BoundRule := .identity (.predecessor 0 63018 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63018 .coefficient)
      LeftBound63016.bound (LeftBound63016.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound63016.derived selector witness)

def rawBound : CoeffClass := LeftBound63016.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound63016.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound63019

namespace LeftBound63025
def owner : Owner := ⟨.program ⟨214⟩, ⟨16140⟩⟩
def transferEvent : Nat := 63025
def frameStart : Nat := 62960
def rule : BoundRule := .product (.predecessor 0 63023 .coefficient) (.predecessor 1 63024 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63023 .coefficient)
      LeftAuthority63021.bound (LeftAuthority63021.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63022RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63021.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63021.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63024 .coefficient)
      LeftBound63019.bound (LeftBound63019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63019.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority63021.bound LeftBound63019.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63021.bound, LeftBound63019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority63021.actual selector witness) * (LeftBound63019.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63025

namespace LeftBound63033
def owner : Owner := ⟨.program ⟨214⟩, ⟨16141⟩⟩
def transferEvent : Nat := 63033
def frameStart : Nat := 62960
def rule : BoundRule := .sum [.predecessor 0 63031 .coefficient, .predecessor 1 63032 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63031 .coefficient)
      LeftAuthority63029.bound (LeftAuthority63029.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63030RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63029.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63029.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63032 .coefficient)
      LeftBound63025.bound (LeftBound63025.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63027RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63025.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63025.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority63029.bound, LeftBound63025.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63029.bound, LeftBound63025.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority63029.actual selector witness, LeftBound63025.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63033

namespace LeftBound63037
def owner : Owner := ⟨.program ⟨214⟩, ⟨28090⟩⟩
def transferEvent : Nat := 63037
def frameStart : Nat := 62960
def rule : BoundRule := .product (.predecessor 0 63035 .coefficient) (.predecessor 1 63036 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63035 .coefficient)
      LeftBound63033.bound (LeftBound63033.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63034RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63033.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63033.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63036 .coefficient)
      LeftAuthority63010.bound (LeftAuthority63010.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63011RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63010.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63010.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound63033.bound LeftAuthority63010.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63033.bound, LeftAuthority63010.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound63033.actual selector witness) * (LeftAuthority63010.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63037

namespace LeftBound63048
def owner : Owner := ⟨.program ⟨214⟩, ⟨18047⟩⟩
def transferEvent : Nat := 63048
def frameStart : Nat := 62960
def rule : BoundRule := .product (.predecessor 0 63046 .coefficient) (.predecessor 1 63047 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63046 .coefficient)
      LeftAuthority63021.bound (LeftAuthority63021.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63022RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63021.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63021.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63047 .coefficient)
      LeftAuthority63044.bound (LeftAuthority63044.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63044.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63044.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority63021.bound LeftAuthority63044.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63021.bound, LeftAuthority63044.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority63021.actual selector witness) * (LeftAuthority63044.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63048

namespace LeftBound63056
def owner : Owner := ⟨.program ⟨214⟩, ⟨18048⟩⟩
def transferEvent : Nat := 63056
def frameStart : Nat := 62960
def rule : BoundRule := .sum [.predecessor 0 63054 .coefficient, .predecessor 1 63055 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63054 .coefficient)
      LeftAuthority63052.bound (LeftAuthority63052.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63053RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63052.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63052.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63055 .coefficient)
      LeftBound63048.bound (LeftBound63048.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63050RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63048.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63048.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority63052.bound, LeftBound63048.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63052.bound, LeftBound63048.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority63052.actual selector witness, LeftBound63048.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63056

namespace LeftBound63060
def owner : Owner := ⟨.program ⟨214⟩, ⟨28095⟩⟩
def transferEvent : Nat := 63060
def frameStart : Nat := 62960
def rule : BoundRule := .sum [.predecessor 0 63058 .coefficient, .predecessor 1 63059 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63058 .coefficient)
      LeftBound63056.bound (LeftBound63056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63056.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63056.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63059 .coefficient)
      LeftBound63037.bound (LeftBound63037.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63042RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63037.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63037.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63056.bound, LeftBound63037.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63056.bound, LeftBound63037.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound63056.actual selector witness, LeftBound63037.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63060

namespace LeftBound63073
def owner : Owner := ⟨.program ⟨214⟩, ⟨28092⟩⟩
def transferEvent : Nat := 63073
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 63071 .coefficient, .predecessor 1 63072 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63071 .coefficient)
      LeftBound62902.bound (LeftBound62902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63070RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62902.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62902.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63072 .coefficient)
      LeftBound62885.bound (LeftBound62885.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events245.exact62892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62885.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62885.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62902.bound, LeftBound62885.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62902.bound, LeftBound62885.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound62902.actual selector witness, LeftBound62885.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63073

namespace LeftBound63076
def owner : Owner := ⟨.program ⟨214⟩, ⟨28092⟩⟩
def transferEvent : Nat := 63076
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 63070 .summary, .result 62892 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63070 .summary)
      LeftBound62904.bound (LeftBound62904.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21479⟩⟩) (rawTerms := some (Proof.Events246.exact63070RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62904.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62892 .summary)
      LeftBound62887.bound (LeftBound62887.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28091⟩⟩) (rawTerms := some (Proof.Events245.exact62892RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62887.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62904.bound, LeftBound62887.bound]
def bound : CoeffClass := .finite ⟨1292113298829627502592, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62904.bound, LeftBound62887.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound62904.actual selector witness, LeftBound62887.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63076

namespace LeftBound63080
def owner : Owner := ⟨.program ⟨214⟩, ⟨28093⟩⟩
def transferEvent : Nat := 63080
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 63078 .coefficient) (.predecessor 1 63079 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63078 .coefficient)
      LeftBound63073.bound (LeftBound63073.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63077RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63073.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63073.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63079 .coefficient)
      LeftBound5698.bound (LeftBound5698.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5699RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5698.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5698.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound63073.bound LeftBound5698.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63073.bound, LeftBound5698.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound63073.actual selector witness) * (LeftBound5698.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63080

namespace LeftBound63081
def owner : Owner := ⟨.program ⟨214⟩, ⟨28093⟩⟩
def transferEvent : Nat := 63081
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩ [⟨.result 5695 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5695 .coefficient)
      LeftAuthority5694.bound (LeftAuthority5694.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6637⟩⟩) (rawTerms := some (Proof.Events022.exact5695RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5694.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5694.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5694.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5694.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5694.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound63081

namespace LeftBound63082
def owner : Owner := ⟨.program ⟨214⟩, ⟨28093⟩⟩
def transferEvent : Nat := 63082
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 63077 .summary) (.transfer 63081) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63077 .summary)
      LeftBound63076.bound (LeftBound63076.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28092⟩⟩) (rawTerms := some (Proof.Events246.exact63077RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63076.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 63081)
      LeftBound63081.bound (LeftBound63081.actual selector witness) := by
  exact .transfer (LeftBound63081.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound63076.bound LeftBound63081.bound
def bound : CoeffClass := .finite ⟨4742076480517514208552681472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63076.bound, LeftBound63081.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound63076.actual selector witness) * (LeftBound63081.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63082

namespace LeftBound63097
def owner : Owner := ⟨.program ⟨214⟩, ⟨27874⟩⟩
def transferEvent : Nat := 63097
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 63095 .coefficient) (.predecessor 1 63096 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63095 .coefficient)
      LeftBound55764.bound (LeftBound55764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55768RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55764.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55764.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63096 .coefficient)
      LeftAuthority63093.bound (LeftAuthority63093.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63094RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63093.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63093.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound55764.bound LeftAuthority63093.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55764.bound, LeftAuthority63093.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound55764.actual selector witness) * (LeftAuthority63093.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63097

namespace LeftBound63098
def owner : Owner := ⟨.program ⟨214⟩, ⟨27874⟩⟩
def transferEvent : Nat := 63098
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27872⟩⟩]⟩ [⟨.result 63094 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63094 .coefficient)
      LeftAuthority63093.bound (LeftAuthority63093.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27872⟩⟩) (rawTerms := some (Proof.Events246.exact63094RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63093.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63093.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority63093.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63093.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority63093.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound63098

namespace LeftBound63099
def owner : Owner := ⟨.program ⟨214⟩, ⟨27874⟩⟩
def transferEvent : Nat := 63099
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 55768 .summary) (.transfer 63098) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55768 .summary)
      LeftBound55767.bound (LeftBound55767.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26073⟩⟩) (rawTerms := some (Proof.Events217.exact55768RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55767.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 63098)
      LeftBound63098.bound (LeftBound63098.actual selector witness) := by
  exact .transfer (LeftBound63098.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound55767.bound LeftBound63098.bound
def bound : CoeffClass := .finite ⟨1292068472128282820608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55767.bound, LeftBound63098.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound55767.actual selector witness) * (LeftBound63098.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63099

namespace LeftBound63110
def owner : Owner := ⟨.program ⟨214⟩, ⟨21334⟩⟩
def transferEvent : Nat := 63110
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 63108 .coefficient) (.value (.predecessor 1 63109 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63108 .coefficient)
      LeftAuthority63106.bound (LeftAuthority63106.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63107RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63106.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63106.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63109 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority63106.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63106.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority63106.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound63110

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
