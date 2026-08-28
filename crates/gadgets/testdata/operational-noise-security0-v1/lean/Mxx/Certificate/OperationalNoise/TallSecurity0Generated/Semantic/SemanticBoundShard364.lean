import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard057
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard363

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound54018
def owner : Owner := ⟨.program ⟨214⟩, ⟨28752⟩⟩
def transferEvent : Nat := 54018
def frameStart : Nat := 53918
def rule : BoundRule := .sum [.predecessor 0 54016 .coefficient, .predecessor 1 54017 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54016 .coefficient)
      LeftBound54014.bound (LeftBound54014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact54015RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54014.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54014.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54017 .coefficient)
      LeftBound53995.bound (LeftBound53995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact54000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53995.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53995.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54014.bound, LeftBound53995.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54014.bound, LeftBound53995.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54014.actual selector witness, LeftBound53995.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54018

namespace LeftBound54031
def owner : Owner := ⟨.program ⟨214⟩, ⟨28750⟩⟩
def transferEvent : Nat := 54031
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54029 .coefficient, .predecessor 1 54030 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54029 .coefficient)
      LeftBound53860.bound (LeftBound53860.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53860.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53860.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54030 .coefficient)
      LeftBound53843.bound (LeftBound53843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53850RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53843.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53843.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53860.bound, LeftBound53843.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53860.bound, LeftBound53843.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound53860.actual selector witness, LeftBound53843.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54031

namespace LeftBound54034
def owner : Owner := ⟨.program ⟨214⟩, ⟨28750⟩⟩
def transferEvent : Nat := 54034
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 54028 .summary, .result 53850 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54028 .summary)
      LeftBound53862.bound (LeftBound53862.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21983⟩⟩) (rawTerms := some (Proof.Events211.exact54028RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53862.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53850 .summary)
      LeftBound53845.bound (LeftBound53845.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28749⟩⟩) (rawTerms := some (Proof.Events210.exact53850RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53845.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53862.bound, LeftBound53845.bound]
def bound : CoeffClass := .finite ⟨1292270185944771604480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53862.bound, LeftBound53845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound53862.actual selector witness, LeftBound53845.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54034

namespace LeftBound54058
def owner : Owner := ⟨.program ⟨214⟩, ⟨11772⟩⟩
def transferEvent : Nat := 54058
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 54056 .coefficient) (.predecessor 1 54057 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54056 .coefficient)
      LeftAuthority2498.bound (LeftAuthority2498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2499RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2498.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2498.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54057 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2498.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2498.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2498.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound54058

namespace LeftBound54063
def owner : Owner := ⟨.program ⟨214⟩, ⟨7277⟩⟩
def transferEvent : Nat := 54063
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 54061 .coefficient) (.predecessor 1 54062 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54061 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54062 .coefficient)
      LeftBound9978.bound (LeftBound9978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9978.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound9978.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound9978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound9978.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54063

namespace LeftBound54068
def owner : Owner := ⟨.program ⟨214⟩, ⟨11773⟩⟩
def transferEvent : Nat := 54068
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54066 .coefficient, .predecessor 1 54067 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54066 .coefficient)
      LeftBound54063.bound (LeftBound54063.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54065RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54063.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54063.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54067 .coefficient)
      LeftBound54058.bound (LeftBound54058.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54060RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54058.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54058.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54063.bound, LeftBound54058.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54063.bound, LeftBound54058.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54063.actual selector witness, LeftBound54058.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54068

namespace LeftBound54072
def owner : Owner := ⟨.program ⟨214⟩, ⟨11774⟩⟩
def transferEvent : Nat := 54072
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54070 .coefficient, .predecessor 1 54071 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54070 .coefficient)
      LeftBound54068.bound (LeftBound54068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54069RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54068.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54068.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54071 .coefficient)
      LeftBound9970.bound (LeftBound9970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9971RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9970.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54068.bound, LeftBound9970.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54068.bound, LeftBound9970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54068.actual selector witness, LeftBound9970.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54072

namespace LeftBound54073
def owner : Owner := ⟨.program ⟨214⟩, ⟨11774⟩⟩
def transferEvent : Nat := 54073
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨97⟩⟩]⟩ [⟨.result 9971 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9971 .coefficient)
      LeftBound9970.bound (LeftBound9970.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨97⟩⟩) (rawTerms := some (Proof.Events038.exact9971RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9970.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound9970.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound9970.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound54073

namespace LeftBound54078
def owner : Owner := ⟨.program ⟨214⟩, ⟨11775⟩⟩
def transferEvent : Nat := 54078
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 54076 .coefficient) (.predecessor 1 54077 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54076 .coefficient)
      LeftBound54072.bound (LeftBound54072.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54072.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54072.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54077 .coefficient)
      LeftAuthority2501.bound (LeftAuthority2501.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2502RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2501.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2501.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound54072.bound LeftAuthority2501.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54072.bound, LeftAuthority2501.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound54072.actual selector witness) * (LeftAuthority2501.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54078

namespace LeftBound54079
def owner : Owner := ⟨.program ⟨214⟩, ⟨11775⟩⟩
def transferEvent : Nat := 54079
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9615⟩⟩], []⟩ [⟨.result 2502 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2502 .coefficient)
      LeftAuthority2501.bound (LeftAuthority2501.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9615⟩⟩) (rawTerms := some (Proof.Events009.exact2502RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2501.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2501.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority2501.bound []
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2501.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority2501.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound54079

namespace LeftBound54080
def owner : Owner := ⟨.program ⟨214⟩, ⟨11775⟩⟩
def transferEvent : Nat := 54080
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 54075 .summary) (.transfer 54079) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54075 .summary)
      LeftBound54073.bound (LeftBound54073.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11774⟩⟩) (rawTerms := some (Proof.Events211.exact54075RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54073.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 54079)
      LeftBound54079.bound (LeftBound54079.actual selector witness) := by
  exact .transfer (LeftBound54079.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound54073.bound LeftBound54079.bound
def bound : CoeffClass := .finite ⟨24960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54073.bound, LeftBound54079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound54073.actual selector witness) * (LeftBound54079.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54080

namespace LeftBound54086
def owner : Owner := ⟨.program ⟨214⟩, ⟨9616⟩⟩
def transferEvent : Nat := 54086
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 54084 .coefficient) (.predecessor 1 54085 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54084 .coefficient)
      LeftAuthority2501.bound (LeftAuthority2501.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2502RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2501.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2501.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54085 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2501.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2501.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2501.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound54086

namespace LeftBound54091
def owner : Owner := ⟨.program ⟨214⟩, ⟨7257⟩⟩
def transferEvent : Nat := 54091
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 54089 .coefficient) (.predecessor 1 54090 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54089 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54090 .coefficient)
      LeftBound10019.bound (LeftBound10019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10019.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound10019.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound10019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound10019.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54091

namespace LeftBound54096
def owner : Owner := ⟨.program ⟨214⟩, ⟨9617⟩⟩
def transferEvent : Nat := 54096
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54094 .coefficient, .predecessor 1 54095 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54094 .coefficient)
      LeftBound54091.bound (LeftBound54091.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54093RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54091.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54091.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54095 .coefficient)
      LeftBound54086.bound (LeftBound54086.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54086.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54086.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54091.bound, LeftBound54086.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54091.bound, LeftBound54086.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54091.actual selector witness, LeftBound54086.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54096

namespace LeftBound54100
def owner : Owner := ⟨.program ⟨214⟩, ⟨9618⟩⟩
def transferEvent : Nat := 54100
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54098 .coefficient, .predecessor 1 54099 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54098 .coefficient)
      LeftBound54096.bound (LeftBound54096.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54097RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54096.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54096.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54099 .coefficient)
      LeftBound10011.bound (LeftBound10011.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10011.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10011.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54096.bound, LeftBound10011.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54096.bound, LeftBound10011.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54096.actual selector witness, LeftBound10011.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54100

namespace LeftBound54101
def owner : Owner := ⟨.program ⟨214⟩, ⟨9618⟩⟩
def transferEvent : Nat := 54101
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨77⟩⟩]⟩ [⟨.result 10012 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10012 .coefficient)
      LeftBound10011.bound (LeftBound10011.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨77⟩⟩) (rawTerms := some (Proof.Events039.exact10012RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10011.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10011.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound10011.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10011.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound10011.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound54101

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
