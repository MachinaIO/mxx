import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard091
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard092
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard497

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound72977
def owner : Owner := ⟨.program ⟨214⟩, ⟨15365⟩⟩
def transferEvent : Nat := 72977
def frameStart : Nat := 72881
def rule : BoundRule := .sum [.predecessor 0 72975 .coefficient, .predecessor 1 72976 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72975 .coefficient)
      LeftAuthority72973.bound (LeftAuthority72973.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact72974RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72973.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72973.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72976 .coefficient)
      LeftBound72969.bound (LeftBound72969.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact72971RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72969.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72969.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority72973.bound, LeftBound72969.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72973.bound, LeftBound72969.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority72973.actual selector witness, LeftBound72969.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72977

namespace LeftBound72981
def owner : Owner := ⟨.program ⟨214⟩, ⟨26773⟩⟩
def transferEvent : Nat := 72981
def frameStart : Nat := 72881
def rule : BoundRule := .sum [.predecessor 0 72979 .coefficient, .predecessor 1 72980 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72979 .coefficient)
      LeftBound72977.bound (LeftBound72977.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact72978RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72977.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72977.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72980 .coefficient)
      LeftBound72958.bound (LeftBound72958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact72963RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72958.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72958.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72977.bound, LeftBound72958.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72977.bound, LeftBound72958.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72977.actual selector witness, LeftBound72958.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72981

namespace LeftBound72994
def owner : Owner := ⟨.program ⟨214⟩, ⟨26771⟩⟩
def transferEvent : Nat := 72994
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 72992 .coefficient, .predecessor 1 72993 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72992 .coefficient)
      LeftBound72823.bound (LeftBound72823.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact72991RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72823.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72823.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72993 .coefficient)
      LeftBound72806.bound (LeftBound72806.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72813RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72806.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72806.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72823.bound, LeftBound72806.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72823.bound, LeftBound72806.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72823.actual selector witness, LeftBound72806.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72994

namespace LeftBound72997
def owner : Owner := ⟨.program ⟨214⟩, ⟨26771⟩⟩
def transferEvent : Nat := 72997
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 72991 .summary, .result 72813 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72991 .summary)
      LeftBound72825.bound (LeftBound72825.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20679⟩⟩) (rawTerms := some (Proof.Events285.exact72991RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72825.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72813 .summary)
      LeftBound72808.bound (LeftBound72808.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26770⟩⟩) (rawTerms := some (Proof.Events284.exact72813RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72808.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72825.bound, LeftBound72808.bound]
def bound : CoeffClass := .finite ⟨1291911586824442228736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72825.bound, LeftBound72808.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72825.actual selector witness, LeftBound72808.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72997

namespace LeftBound73021
def owner : Owner := ⟨.program ⟨214⟩, ⟨10671⟩⟩
def transferEvent : Nat := 73021
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 73019 .coefficient) (.predecessor 1 73020 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73019 .coefficient)
      LeftAuthority3453.bound (LeftAuthority3453.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3454RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3453.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3453.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73020 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3453.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3453.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3453.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound73021

namespace LeftBound73026
def owner : Owner := ⟨.program ⟨214⟩, ⟨7191⟩⟩
def transferEvent : Nat := 73026
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 73024 .coefficient) (.predecessor 1 73025 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73024 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73025 .coefficient)
      LeftBound14487.bound (LeftBound14487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14487.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14487.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound14487.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound14487.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound14487.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73026

namespace LeftBound73031
def owner : Owner := ⟨.program ⟨214⟩, ⟨10672⟩⟩
def transferEvent : Nat := 73031
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 73029 .coefficient, .predecessor 1 73030 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73029 .coefficient)
      LeftBound73026.bound (LeftBound73026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73026.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73030 .coefficient)
      LeftBound73021.bound (LeftBound73021.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73021.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73021.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73026.bound, LeftBound73021.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73026.bound, LeftBound73021.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73026.actual selector witness, LeftBound73021.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73031

namespace LeftBound73035
def owner : Owner := ⟨.program ⟨214⟩, ⟨10673⟩⟩
def transferEvent : Nat := 73035
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 73033 .coefficient, .predecessor 1 73034 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73033 .coefficient)
      LeftBound73031.bound (LeftBound73031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73031.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73031.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73034 .coefficient)
      LeftBound14479.bound (LeftBound14479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14479.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73031.bound, LeftBound14479.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73031.bound, LeftBound14479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73031.actual selector witness, LeftBound14479.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73035

namespace LeftBound73036
def owner : Owner := ⟨.program ⟨214⟩, ⟨10673⟩⟩
def transferEvent : Nat := 73036
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨87⟩⟩]⟩ [⟨.result 14480 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14480 .coefficient)
      LeftBound14479.bound (LeftBound14479.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨87⟩⟩) (rawTerms := some (Proof.Events056.exact14480RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14479.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound14479.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound14479.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound73036

namespace LeftBound73041
def owner : Owner := ⟨.program ⟨214⟩, ⟨10674⟩⟩
def transferEvent : Nat := 73041
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 73039 .coefficient) (.predecessor 1 73040 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73039 .coefficient)
      LeftBound73035.bound (LeftBound73035.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73038RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73035.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73035.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73040 .coefficient)
      LeftAuthority3456.bound (LeftAuthority3456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3456.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3456.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound73035.bound LeftAuthority3456.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73035.bound, LeftAuthority3456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound73035.actual selector witness) * (LeftAuthority3456.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73041

namespace LeftBound73042
def owner : Owner := ⟨.program ⟨214⟩, ⟨10674⟩⟩
def transferEvent : Nat := 73042
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9500⟩⟩], []⟩ [⟨.result 3457 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3457 .coefficient)
      LeftAuthority3456.bound (LeftAuthority3456.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9500⟩⟩) (rawTerms := some (Proof.Events013.exact3457RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3456.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3456.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3456.bound []
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority3456.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound73042

namespace LeftBound73043
def owner : Owner := ⟨.program ⟨214⟩, ⟨10674⟩⟩
def transferEvent : Nat := 73043
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 73038 .summary) (.transfer 73042) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73038 .summary)
      LeftBound73036.bound (LeftBound73036.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10673⟩⟩) (rawTerms := some (Proof.Events285.exact73038RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73036.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 73042)
      LeftBound73042.bound (LeftBound73042.actual selector witness) := by
  exact .transfer (LeftBound73042.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound73036.bound LeftBound73042.bound
def bound : CoeffClass := .finite ⟨2496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73036.bound, LeftBound73042.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound73036.actual selector witness) * (LeftBound73042.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73043

namespace LeftBound73049
def owner : Owner := ⟨.program ⟨214⟩, ⟨9501⟩⟩
def transferEvent : Nat := 73049
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 73047 .coefficient) (.predecessor 1 73048 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73047 .coefficient)
      LeftAuthority3456.bound (LeftAuthority3456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3456.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3456.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73048 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3456.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3456.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3456.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound73049

namespace LeftBound73054
def owner : Owner := ⟨.program ⟨214⟩, ⟨7200⟩⟩
def transferEvent : Nat := 73054
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 73052 .coefficient) (.predecessor 1 73053 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73052 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73053 .coefficient)
      LeftBound14528.bound (LeftBound14528.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14529RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14528.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14528.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound14528.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound14528.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound14528.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73054

namespace LeftBound73059
def owner : Owner := ⟨.program ⟨214⟩, ⟨9502⟩⟩
def transferEvent : Nat := 73059
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 73057 .coefficient, .predecessor 1 73058 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73057 .coefficient)
      LeftBound73054.bound (LeftBound73054.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73056RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73054.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73054.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73058 .coefficient)
      LeftBound73049.bound (LeftBound73049.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73051RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73049.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73049.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73054.bound, LeftBound73049.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73054.bound, LeftBound73049.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73054.actual selector witness, LeftBound73049.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73059

namespace LeftBound73063
def owner : Owner := ⟨.program ⟨214⟩, ⟨9503⟩⟩
def transferEvent : Nat := 73063
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 73061 .coefficient, .predecessor 1 73062 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73061 .coefficient)
      LeftBound73059.bound (LeftBound73059.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73060RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73059.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73059.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73062 .coefficient)
      LeftBound14520.bound (LeftBound14520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14520.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73059.bound, LeftBound14520.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73059.bound, LeftBound14520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73059.actual selector witness, LeftBound14520.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73063

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
