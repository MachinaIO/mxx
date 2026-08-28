import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard068
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard237
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard272

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound40795
def owner : Owner := ⟨.program ⟨214⟩, ⟨16142⟩⟩
def transferEvent : Nat := 40795
def frameStart : Nat := 40739
def rule : BoundRule := .sum [.predecessor 0 40793 .coefficient, .predecessor 1 40794 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40793 .coefficient)
      LeftBound40778.bound (LeftBound40778.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound40778.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40794 .coefficient)
      LeftAuthority40791.bound (LeftAuthority40791.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority40791.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40778.bound, LeftAuthority40791.bound]
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40778.bound, LeftAuthority40791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40778.actual selector witness, LeftAuthority40791.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40795

namespace LeftBound40798
def owner : Owner := ⟨.program ⟨214⟩, ⟨16143⟩⟩
def transferEvent : Nat := 40798
def frameStart : Nat := 40739
def rule : BoundRule := .identity (.predecessor 0 40797 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40797 .coefficient)
      LeftBound40795.bound (LeftBound40795.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound40795.derived selector witness)

def rawBound : CoeffClass := LeftBound40795.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40795.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound40795.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound40798

namespace LeftBound40804
def owner : Owner := ⟨.program ⟨214⟩, ⟨16144⟩⟩
def transferEvent : Nat := 40804
def frameStart : Nat := 40739
def rule : BoundRule := .product (.predecessor 0 40802 .coefficient) (.predecessor 1 40803 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40802 .coefficient)
      LeftAuthority40800.bound (LeftAuthority40800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40801RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40800.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40800.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40803 .coefficient)
      LeftBound40798.bound (LeftBound40798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40798.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40798.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority40800.bound LeftBound40798.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40800.bound, LeftBound40798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority40800.actual selector witness) * (LeftBound40798.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40804

namespace LeftBound40812
def owner : Owner := ⟨.program ⟨214⟩, ⟨16145⟩⟩
def transferEvent : Nat := 40812
def frameStart : Nat := 40739
def rule : BoundRule := .sum [.predecessor 0 40810 .coefficient, .predecessor 1 40811 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40810 .coefficient)
      LeftAuthority40808.bound (LeftAuthority40808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40809RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40808.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40808.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40811 .coefficient)
      LeftBound40804.bound (LeftBound40804.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40806RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40804.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40804.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority40808.bound, LeftBound40804.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40808.bound, LeftBound40804.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority40808.actual selector witness, LeftBound40804.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40812

namespace LeftBound40816
def owner : Owner := ⟨.program ⟨214⟩, ⟨28110⟩⟩
def transferEvent : Nat := 40816
def frameStart : Nat := 40739
def rule : BoundRule := .product (.predecessor 0 40814 .coefficient) (.predecessor 1 40815 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40814 .coefficient)
      LeftBound40812.bound (LeftBound40812.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40813RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40812.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40812.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40815 .coefficient)
      LeftAuthority40789.bound (LeftAuthority40789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40789.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40789.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40812.bound LeftAuthority40789.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40812.bound, LeftAuthority40789.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40812.actual selector witness) * (LeftAuthority40789.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40816

namespace LeftBound40827
def owner : Owner := ⟨.program ⟨214⟩, ⟨16112⟩⟩
def transferEvent : Nat := 40827
def frameStart : Nat := 40739
def rule : BoundRule := .product (.predecessor 0 40825 .coefficient) (.predecessor 1 40826 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40825 .coefficient)
      LeftAuthority40800.bound (LeftAuthority40800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40801RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40800.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40800.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40826 .coefficient)
      LeftAuthority40823.bound (LeftAuthority40823.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40824RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40823.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40823.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority40800.bound LeftAuthority40823.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40800.bound, LeftAuthority40823.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority40800.actual selector witness) * (LeftAuthority40823.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40827

namespace LeftBound40835
def owner : Owner := ⟨.program ⟨214⟩, ⟨16113⟩⟩
def transferEvent : Nat := 40835
def frameStart : Nat := 40739
def rule : BoundRule := .sum [.predecessor 0 40833 .coefficient, .predecessor 1 40834 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40833 .coefficient)
      LeftAuthority40831.bound (LeftAuthority40831.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40832RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40831.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40831.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40834 .coefficient)
      LeftBound40827.bound (LeftBound40827.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40829RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40827.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40827.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority40831.bound, LeftBound40827.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40831.bound, LeftBound40827.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority40831.actual selector witness, LeftBound40827.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40835

namespace LeftBound40839
def owner : Owner := ⟨.program ⟨214⟩, ⟨28114⟩⟩
def transferEvent : Nat := 40839
def frameStart : Nat := 40739
def rule : BoundRule := .sum [.predecessor 0 40837 .coefficient, .predecessor 1 40838 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40837 .coefficient)
      LeftBound40835.bound (LeftBound40835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40835.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40835.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40838 .coefficient)
      LeftBound40816.bound (LeftBound40816.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40821RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40816.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40816.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40835.bound, LeftBound40816.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40835.bound, LeftBound40816.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40835.actual selector witness, LeftBound40816.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40839

namespace LeftBound40852
def owner : Owner := ⟨.program ⟨214⟩, ⟨28112⟩⟩
def transferEvent : Nat := 40852
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40850 .coefficient, .predecessor 1 40851 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40850 .coefficient)
      LeftBound40681.bound (LeftBound40681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40849RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40681.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40681.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40851 .coefficient)
      LeftBound40664.bound (LeftBound40664.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40671RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40664.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40664.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40681.bound, LeftBound40664.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40681.bound, LeftBound40664.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40681.actual selector witness, LeftBound40664.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40852

namespace LeftBound40855
def owner : Owner := ⟨.program ⟨214⟩, ⟨28112⟩⟩
def transferEvent : Nat := 40855
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 40849 .summary, .result 40671 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40849 .summary)
      LeftBound40683.bound (LeftBound40683.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21555⟩⟩) (rawTerms := some (Proof.Events159.exact40849RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40683.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40671 .summary)
      LeftBound40666.bound (LeftBound40666.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28111⟩⟩) (rawTerms := some (Proof.Events158.exact40671RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40666.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40683.bound, LeftBound40666.bound]
def bound : CoeffClass := .finite ⟨1292113298829627502592, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40683.bound, LeftBound40666.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40683.actual selector witness, LeftBound40666.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40855

namespace LeftBound40879
def owner : Owner := ⟨.program ⟨214⟩, ⟨11478⟩⟩
def transferEvent : Nat := 40879
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 40877 .coefficient) (.predecessor 1 40878 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40877 .coefficient)
      LeftAuthority1819.bound (LeftAuthority1819.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events007.exact1820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1819.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1819.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40878 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1819.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1819.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1819.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound40879

namespace LeftBound40884
def owner : Owner := ⟨.program ⟨214⟩, ⟨7311⟩⟩
def transferEvent : Nat := 40884
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 40882 .coefficient) (.predecessor 1 40883 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40882 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40883 .coefficient)
      LeftBound11481.bound (LeftBound11481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11481.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11481.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound11481.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound11481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound11481.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40884

namespace LeftBound40889
def owner : Owner := ⟨.program ⟨214⟩, ⟨11479⟩⟩
def transferEvent : Nat := 40889
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40887 .coefficient, .predecessor 1 40888 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40887 .coefficient)
      LeftBound40884.bound (LeftBound40884.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40886RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40884.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40884.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40888 .coefficient)
      LeftBound40879.bound (LeftBound40879.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40881RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40879.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40879.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40884.bound, LeftBound40879.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40884.bound, LeftBound40879.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40884.actual selector witness, LeftBound40879.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40889

namespace LeftBound40893
def owner : Owner := ⟨.program ⟨214⟩, ⟨11480⟩⟩
def transferEvent : Nat := 40893
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40891 .coefficient, .predecessor 1 40892 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40891 .coefficient)
      LeftBound40889.bound (LeftBound40889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40890RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40889.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40889.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40892 .coefficient)
      LeftBound11473.bound (LeftBound11473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11473.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40889.bound, LeftBound11473.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40889.bound, LeftBound11473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40889.actual selector witness, LeftBound11473.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40893

namespace LeftBound40894
def owner : Owner := ⟨.program ⟨214⟩, ⟨11480⟩⟩
def transferEvent : Nat := 40894
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨93⟩⟩]⟩ [⟨.result 11474 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11474 .coefficient)
      LeftBound11473.bound (LeftBound11473.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨93⟩⟩) (rawTerms := some (Proof.Events044.exact11474RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11473.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound11473.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound11473.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound40894

namespace LeftBound40899
def owner : Owner := ⟨.program ⟨214⟩, ⟨14228⟩⟩
def transferEvent : Nat := 40899
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 40897 .coefficient) (.predecessor 1 40898 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40897 .coefficient)
      LeftBound40893.bound (LeftBound40893.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40893.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40893.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40898 .coefficient)
      LeftAuthority1822.bound (LeftAuthority1822.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events007.exact1823RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1822.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1822.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound40893.bound LeftAuthority1822.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40893.bound, LeftAuthority1822.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound40893.actual selector witness) * (LeftAuthority1822.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40899

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
