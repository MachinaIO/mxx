import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard040
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard041
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard146

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound22894
def owner : Owner := ⟨.program ⟨214⟩, ⟨12791⟩⟩
def transferEvent : Nat := 22894
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 22892 .coefficient, .predecessor 1 22893 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22892 .coefficient)
      LeftBound22890.bound (LeftBound22890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22890.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22890.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22893 .coefficient)
      LeftBound7966.bound (LeftBound7966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact7967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7966.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22890.bound, LeftBound7966.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22890.bound, LeftBound7966.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22890.actual selector witness, LeftBound7966.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22894

namespace LeftBound22895
def owner : Owner := ⟨.program ⟨214⟩, ⟨12791⟩⟩
def transferEvent : Nat := 22895
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨101⟩⟩]⟩ [⟨.result 7967 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7967 .coefficient)
      LeftBound7966.bound (LeftBound7966.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨101⟩⟩) (rawTerms := some (Proof.Events031.exact7967RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7966.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound7966.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7966.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound7966.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound22895

namespace LeftBound22900
def owner : Owner := ⟨.program ⟨214⟩, ⟨12792⟩⟩
def transferEvent : Nat := 22900
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 22898 .coefficient) (.predecessor 1 22899 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22898 .coefficient)
      LeftBound22894.bound (LeftBound22894.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22897RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22894.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22894.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22899 .coefficient)
      LeftAuthority913.bound (LeftAuthority913.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority913.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority913.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound22894.bound LeftAuthority913.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22894.bound, LeftAuthority913.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound22894.actual selector witness) * (LeftAuthority913.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22900

namespace LeftBound22901
def owner : Owner := ⟨.program ⟨214⟩, ⟨12792⟩⟩
def transferEvent : Nat := 22901
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨10045⟩⟩], []⟩ [⟨.result 914 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 914 .coefficient)
      LeftAuthority913.bound (LeftAuthority913.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨10045⟩⟩) (rawTerms := some (Proof.Events003.exact914RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority913.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority913.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority913.bound []
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority913.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority913.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound22901

namespace LeftBound22902
def owner : Owner := ⟨.program ⟨214⟩, ⟨12792⟩⟩
def transferEvent : Nat := 22902
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 22897 .summary) (.transfer 22901) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22897 .summary)
      LeftBound22895.bound (LeftBound22895.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12791⟩⟩) (rawTerms := some (Proof.Events089.exact22897RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22895.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 22901)
      LeftBound22901.bound (LeftBound22901.actual selector witness) := by
  exact .transfer (LeftBound22901.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound22895.bound LeftBound22901.bound
def bound : CoeffClass := .finite ⟨38272, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22895.bound, LeftBound22901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound22895.actual selector witness) * (LeftBound22901.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22902

namespace LeftBound22908
def owner : Owner := ⟨.program ⟨214⟩, ⟨10046⟩⟩
def transferEvent : Nat := 22908
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 22906 .coefficient) (.predecessor 1 22907 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22906 .coefficient)
      LeftAuthority913.bound (LeftAuthority913.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority913.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority913.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22907 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority913.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority913.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority913.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound22908

namespace LeftBound22913
def owner : Owner := ⟨.program ⟨214⟩, ⟨7337⟩⟩
def transferEvent : Nat := 22913
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 22911 .coefficient) (.predecessor 1 22912 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22911 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22912 .coefficient)
      LeftBound8015.bound (LeftBound8015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8015.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound8015.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound8015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound8015.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22913

namespace LeftBound22918
def owner : Owner := ⟨.program ⟨214⟩, ⟨10047⟩⟩
def transferEvent : Nat := 22918
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 22916 .coefficient, .predecessor 1 22917 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22916 .coefficient)
      LeftBound22913.bound (LeftBound22913.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22913.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22913.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22917 .coefficient)
      LeftBound22908.bound (LeftBound22908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22910RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22908.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22908.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22913.bound, LeftBound22908.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22913.bound, LeftBound22908.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22913.actual selector witness, LeftBound22908.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22918

namespace LeftBound22922
def owner : Owner := ⟨.program ⟨214⟩, ⟨10048⟩⟩
def transferEvent : Nat := 22922
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 22920 .coefficient, .predecessor 1 22921 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22920 .coefficient)
      LeftBound22918.bound (LeftBound22918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22919RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22918.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22921 .coefficient)
      LeftBound8007.bound (LeftBound8007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8008RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8007.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22918.bound, LeftBound8007.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22918.bound, LeftBound8007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22918.actual selector witness, LeftBound8007.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22922

namespace LeftBound22923
def owner : Owner := ⟨.program ⟨214⟩, ⟨10048⟩⟩
def transferEvent : Nat := 22923
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨81⟩⟩]⟩ [⟨.result 8008 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8008 .coefficient)
      LeftBound8007.bound (LeftBound8007.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨81⟩⟩) (rawTerms := some (Proof.Events031.exact8008RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8007.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound8007.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound8007.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound22923

namespace LeftBound22928
def owner : Owner := ⟨.program ⟨214⟩, ⟨10049⟩⟩
def transferEvent : Nat := 22928
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 22926 .coefficient) (.predecessor 1 22927 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22926 .coefficient)
      LeftBound22922.bound (LeftBound22922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22922.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22922.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22927 .coefficient)
      LeftBound8004.bound (LeftBound8004.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8004.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8004.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound22922.bound LeftBound8004.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22922.bound, LeftBound8004.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound22922.actual selector witness) * (LeftBound8004.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22928

namespace LeftBound22929
def owner : Owner := ⟨.program ⟨214⟩, ⟨10049⟩⟩
def transferEvent : Nat := 22929
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩ [⟨.result 8001 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8001 .coefficient)
      LeftAuthority8000.bound (LeftAuthority8000.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7873⟩⟩) (rawTerms := some (Proof.Events031.exact8001RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8000.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8000.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority8000.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority8000.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound22929

namespace LeftBound22930
def owner : Owner := ⟨.program ⟨214⟩, ⟨10049⟩⟩
def transferEvent : Nat := 22930
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 22925 .summary) (.transfer 22929) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22925 .summary)
      LeftBound22923.bound (LeftBound22923.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10048⟩⟩) (rawTerms := some (Proof.Events089.exact22925RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22923.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 22929)
      LeftBound22929.bound (LeftBound22929.actual selector witness) := by
  exact .transfer (LeftBound22929.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound22923.bound LeftBound22929.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22923.bound, LeftBound22929.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound22923.actual selector witness) * (LeftBound22929.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22930

namespace LeftBound22938
def owner : Owner := ⟨.program ⟨214⟩, ⟨12793⟩⟩
def transferEvent : Nat := 22938
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 22936 .coefficient, .predecessor 1 22937 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22936 .coefficient)
      LeftBound22928.bound (LeftBound22928.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22935RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22928.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22928.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22937 .coefficient)
      LeftBound22900.bound (LeftBound22900.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22905RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22900.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22900.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22928.bound, LeftBound22900.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22928.bound, LeftBound22900.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22928.actual selector witness, LeftBound22900.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22938

namespace LeftBound22940
def owner : Owner := ⟨.program ⟨214⟩, ⟨12793⟩⟩
def transferEvent : Nat := 22940
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 22935 .summary, .result 22905 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22935 .summary)
      LeftBound22930.bound (LeftBound22930.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10049⟩⟩) (rawTerms := some (Proof.Events089.exact22935RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22930.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22905 .summary)
      LeftBound22902.bound (LeftBound22902.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12792⟩⟩) (rawTerms := some (Proof.Events089.exact22905RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22902.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22930.bound, LeftBound22902.bound]
def bound : CoeffClass := .finite ⟨95458688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22930.bound, LeftBound22902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22930.actual selector witness, LeftBound22902.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22940

namespace LeftBound22944
def owner : Owner := ⟨.program ⟨214⟩, ⟨25543⟩⟩
def transferEvent : Nat := 22944
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 22942 .coefficient) (.predecessor 1 22943 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22942 .coefficient)
      LeftBound22938.bound (LeftBound22938.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22941RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22938.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22938.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22943 .coefficient)
      LeftAuthority22876.bound (LeftAuthority22876.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22877RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22876.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22876.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound22938.bound LeftAuthority22876.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22938.bound, LeftAuthority22876.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound22938.actual selector witness) * (LeftAuthority22876.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22944

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
