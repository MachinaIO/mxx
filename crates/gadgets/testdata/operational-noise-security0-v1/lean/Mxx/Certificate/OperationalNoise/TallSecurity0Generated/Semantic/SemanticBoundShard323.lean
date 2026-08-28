import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard283
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard322

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound48703
def owner : Owner := ⟨.program ⟨214⟩, ⟨21195⟩⟩
def transferEvent : Nat := 48703
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 48702) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 48702)
      LeftBound48702.bound (LeftBound48702.actual selector witness) := by
  exact .transfer (LeftBound48702.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound48702.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound48702.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound48702.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48703

namespace LeftBound48798
def owner : Owner := ⟨.program ⟨214⟩, ⟨15830⟩⟩
def transferEvent : Nat := 48798
def frameStart : Nat := 48759
def rule : BoundRule := .identity (.predecessor 0 48797 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48797 .coefficient)
      LeftAuthority48795.bound (LeftAuthority48795.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48796RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48795.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48795.derived selector witness)

def rawBound : CoeffClass := LeftAuthority48795.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48795.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority48795.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound48798

namespace LeftBound48815
def owner : Owner := ⟨.program ⟨214⟩, ⟨15904⟩⟩
def transferEvent : Nat := 48815
def frameStart : Nat := 48759
def rule : BoundRule := .sum [.predecessor 0 48813 .coefficient, .predecessor 1 48814 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48813 .coefficient)
      LeftBound48798.bound (LeftBound48798.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound48798.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48814 .coefficient)
      LeftAuthority48811.bound (LeftAuthority48811.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority48811.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48798.bound, LeftAuthority48811.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48798.bound, LeftAuthority48811.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound48798.actual selector witness, LeftAuthority48811.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48815

namespace LeftBound48818
def owner : Owner := ⟨.program ⟨214⟩, ⟨15905⟩⟩
def transferEvent : Nat := 48818
def frameStart : Nat := 48759
def rule : BoundRule := .identity (.predecessor 0 48817 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48817 .coefficient)
      LeftBound48815.bound (LeftBound48815.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound48815.derived selector witness)

def rawBound : CoeffClass := LeftBound48815.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48815.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound48815.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound48818

namespace LeftBound48824
def owner : Owner := ⟨.program ⟨214⟩, ⟨15906⟩⟩
def transferEvent : Nat := 48824
def frameStart : Nat := 48759
def rule : BoundRule := .product (.predecessor 0 48822 .coefficient) (.predecessor 1 48823 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48822 .coefficient)
      LeftAuthority48820.bound (LeftAuthority48820.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48821RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48820.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48820.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48823 .coefficient)
      LeftBound48818.bound (LeftBound48818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48819RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48818.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48818.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority48820.bound LeftBound48818.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48820.bound, LeftBound48818.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority48820.actual selector witness) * (LeftBound48818.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48824

namespace LeftBound48832
def owner : Owner := ⟨.program ⟨214⟩, ⟨15907⟩⟩
def transferEvent : Nat := 48832
def frameStart : Nat := 48759
def rule : BoundRule := .sum [.predecessor 0 48830 .coefficient, .predecessor 1 48831 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48830 .coefficient)
      LeftAuthority48828.bound (LeftAuthority48828.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48829RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48828.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48828.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48831 .coefficient)
      LeftBound48824.bound (LeftBound48824.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48826RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48824.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48824.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority48828.bound, LeftBound48824.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48828.bound, LeftBound48824.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority48828.actual selector witness, LeftBound48824.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48832

namespace LeftBound48836
def owner : Owner := ⟨.program ⟨214⟩, ⟨27669⟩⟩
def transferEvent : Nat := 48836
def frameStart : Nat := 48759
def rule : BoundRule := .product (.predecessor 0 48834 .coefficient) (.predecessor 1 48835 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48834 .coefficient)
      LeftBound48832.bound (LeftBound48832.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48833RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48832.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48832.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48835 .coefficient)
      LeftAuthority48809.bound (LeftAuthority48809.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48810RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48809.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48809.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound48832.bound LeftAuthority48809.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48832.bound, LeftAuthority48809.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound48832.actual selector witness) * (LeftAuthority48809.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48836

namespace LeftBound48847
def owner : Owner := ⟨.program ⟨214⟩, ⟨17231⟩⟩
def transferEvent : Nat := 48847
def frameStart : Nat := 48759
def rule : BoundRule := .product (.predecessor 0 48845 .coefficient) (.predecessor 1 48846 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48845 .coefficient)
      LeftAuthority48820.bound (LeftAuthority48820.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48821RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48820.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48820.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48846 .coefficient)
      LeftAuthority48843.bound (LeftAuthority48843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48844RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48843.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48843.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority48820.bound LeftAuthority48843.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48820.bound, LeftAuthority48843.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority48820.actual selector witness) * (LeftAuthority48843.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48847

namespace LeftBound48855
def owner : Owner := ⟨.program ⟨214⟩, ⟨17232⟩⟩
def transferEvent : Nat := 48855
def frameStart : Nat := 48759
def rule : BoundRule := .sum [.predecessor 0 48853 .coefficient, .predecessor 1 48854 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48853 .coefficient)
      LeftAuthority48851.bound (LeftAuthority48851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48851.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48851.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48854 .coefficient)
      LeftBound48847.bound (LeftBound48847.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48849RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48847.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48847.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority48851.bound, LeftBound48847.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48851.bound, LeftBound48847.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority48851.actual selector witness, LeftBound48847.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48855

namespace LeftBound48859
def owner : Owner := ⟨.program ⟨214⟩, ⟨27674⟩⟩
def transferEvent : Nat := 48859
def frameStart : Nat := 48759
def rule : BoundRule := .sum [.predecessor 0 48857 .coefficient, .predecessor 1 48858 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48857 .coefficient)
      LeftBound48855.bound (LeftBound48855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48855.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48855.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48858 .coefficient)
      LeftBound48836.bound (LeftBound48836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48841RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48836.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48836.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48855.bound, LeftBound48836.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48855.bound, LeftBound48836.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound48855.actual selector witness, LeftBound48836.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48859

namespace LeftBound48872
def owner : Owner := ⟨.program ⟨214⟩, ⟨27671⟩⟩
def transferEvent : Nat := 48872
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 48870 .coefficient, .predecessor 1 48871 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48870 .coefficient)
      LeftBound48701.bound (LeftBound48701.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48701.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48701.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48871 .coefficient)
      LeftBound48684.bound (LeftBound48684.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48691RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48684.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48684.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48701.bound, LeftBound48684.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48701.bound, LeftBound48684.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound48701.actual selector witness, LeftBound48684.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48872

namespace LeftBound48875
def owner : Owner := ⟨.program ⟨214⟩, ⟨27671⟩⟩
def transferEvent : Nat := 48875
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 48869 .summary, .result 48691 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48869 .summary)
      LeftBound48703.bound (LeftBound48703.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21195⟩⟩) (rawTerms := some (Proof.Events190.exact48869RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound48703.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48691 .summary)
      LeftBound48686.bound (LeftBound48686.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27670⟩⟩) (rawTerms := some (Proof.Events190.exact48691RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound48686.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48703.bound, LeftBound48686.bound]
def bound : CoeffClass := .finite ⟨1292046061494565744640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48703.bound, LeftBound48686.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound48703.actual selector witness, LeftBound48686.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48875

namespace LeftBound48879
def owner : Owner := ⟨.program ⟨214⟩, ⟨27672⟩⟩
def transferEvent : Nat := 48879
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 48877 .coefficient) (.predecessor 1 48878 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48877 .coefficient)
      LeftBound48872.bound (LeftBound48872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48872.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48878 .coefficient)
      LeftBound5738.bound (LeftBound5738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5739RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5738.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5738.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound48872.bound LeftBound5738.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48872.bound, LeftBound5738.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound48872.actual selector witness) * (LeftBound5738.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48879

namespace LeftBound48880
def owner : Owner := ⟨.program ⟨214⟩, ⟨27672⟩⟩
def transferEvent : Nat := 48880
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6643⟩⟩]⟩ [⟨.result 5735 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5735 .coefficient)
      LeftAuthority5734.bound (LeftAuthority5734.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6643⟩⟩) (rawTerms := some (Proof.Events022.exact5735RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5734.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5734.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5734.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5734.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5734.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound48880

namespace LeftBound48881
def owner : Owner := ⟨.program ⟨214⟩, ⟨27672⟩⟩
def transferEvent : Nat := 48881
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 48876 .summary) (.transfer 48880) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48876 .summary)
      LeftBound48875.bound (LeftBound48875.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27671⟩⟩) (rawTerms := some (Proof.Events190.exact48876RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound48875.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 48880)
      LeftBound48880.bound (LeftBound48880.actual selector witness) := by
  exact .transfer (LeftBound48880.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound48875.bound LeftBound48880.bound
def bound : CoeffClass := .finite ⟨4741829718422040195880714240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48875.bound, LeftBound48880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound48875.actual selector witness) * (LeftBound48880.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48881

namespace LeftBound48896
def owner : Owner := ⟨.program ⟨214⟩, ⟨27453⟩⟩
def transferEvent : Nat := 48896
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 48894 .coefficient) (.predecessor 1 48895 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48894 .coefficient)
      LeftBound42103.bound (LeftBound42103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42107RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42103.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42103.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48895 .coefficient)
      LeftAuthority48892.bound (LeftAuthority48892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48892.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48892.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound42103.bound LeftAuthority48892.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42103.bound, LeftAuthority48892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound42103.actual selector witness) * (LeftAuthority48892.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48896

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
