import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard036
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard446

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound66198
def owner : Owner := ⟨.program ⟨214⟩, ⟨16965⟩⟩
def transferEvent : Nat := 66198
def frameStart : Nat := 66133
def rule : BoundRule := .product (.predecessor 0 66196 .coefficient) (.predecessor 1 66197 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66196 .coefficient)
      LeftAuthority66194.bound (LeftAuthority66194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66194.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66194.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66197 .coefficient)
      LeftBound66192.bound (LeftBound66192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66192.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority66194.bound LeftBound66192.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66194.bound, LeftBound66192.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority66194.actual selector witness) * (LeftBound66192.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66198

namespace LeftBound66206
def owner : Owner := ⟨.program ⟨214⟩, ⟨16966⟩⟩
def transferEvent : Nat := 66206
def frameStart : Nat := 66133
def rule : BoundRule := .sum [.predecessor 0 66204 .coefficient, .predecessor 1 66205 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66204 .coefficient)
      LeftAuthority66202.bound (LeftAuthority66202.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66203RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66202.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66202.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66205 .coefficient)
      LeftBound66198.bound (LeftBound66198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66200RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66198.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66198.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority66202.bound, LeftBound66198.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66202.bound, LeftBound66198.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority66202.actual selector witness, LeftBound66198.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66206

namespace LeftBound66210
def owner : Owner := ⟨.program ⟨214⟩, ⟨29807⟩⟩
def transferEvent : Nat := 66210
def frameStart : Nat := 66133
def rule : BoundRule := .product (.predecessor 0 66208 .coefficient) (.predecessor 1 66209 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66208 .coefficient)
      LeftBound66206.bound (LeftBound66206.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66207RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66206.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66206.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66209 .coefficient)
      LeftAuthority66183.bound (LeftAuthority66183.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66184RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66183.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66183.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66206.bound LeftAuthority66183.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66206.bound, LeftAuthority66183.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66206.actual selector witness) * (LeftAuthority66183.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66210

namespace LeftBound66221
def owner : Owner := ⟨.program ⟨214⟩, ⟨17083⟩⟩
def transferEvent : Nat := 66221
def frameStart : Nat := 66133
def rule : BoundRule := .product (.predecessor 0 66219 .coefficient) (.predecessor 1 66220 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66219 .coefficient)
      LeftAuthority66194.bound (LeftAuthority66194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66194.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66194.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66220 .coefficient)
      LeftAuthority66217.bound (LeftAuthority66217.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66217.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66217.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority66194.bound LeftAuthority66217.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66194.bound, LeftAuthority66217.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority66194.actual selector witness) * (LeftAuthority66217.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66221

namespace LeftBound66229
def owner : Owner := ⟨.program ⟨214⟩, ⟨17084⟩⟩
def transferEvent : Nat := 66229
def frameStart : Nat := 66133
def rule : BoundRule := .sum [.predecessor 0 66227 .coefficient, .predecessor 1 66228 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66227 .coefficient)
      LeftAuthority66225.bound (LeftAuthority66225.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66226RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66225.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66225.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66228 .coefficient)
      LeftBound66221.bound (LeftBound66221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66221.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66221.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority66225.bound, LeftBound66221.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66225.bound, LeftBound66221.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority66225.actual selector witness, LeftBound66221.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66229

namespace LeftBound66233
def owner : Owner := ⟨.program ⟨214⟩, ⟨29811⟩⟩
def transferEvent : Nat := 66233
def frameStart : Nat := 66133
def rule : BoundRule := .sum [.predecessor 0 66231 .coefficient, .predecessor 1 66232 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66231 .coefficient)
      LeftBound66229.bound (LeftBound66229.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66230RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66229.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66229.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66232 .coefficient)
      LeftBound66210.bound (LeftBound66210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66210.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66210.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66229.bound, LeftBound66210.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66229.bound, LeftBound66210.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66229.actual selector witness, LeftBound66210.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66233

namespace LeftBound66246
def owner : Owner := ⟨.program ⟨214⟩, ⟨29809⟩⟩
def transferEvent : Nat := 66246
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 66244 .coefficient, .predecessor 1 66245 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66244 .coefficient)
      LeftBound66075.bound (LeftBound66075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66243RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66075.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66075.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66245 .coefficient)
      LeftBound66058.bound (LeftBound66058.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66065RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66058.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66058.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66075.bound, LeftBound66058.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66075.bound, LeftBound66058.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66075.actual selector witness, LeftBound66058.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66246

namespace LeftBound66249
def owner : Owner := ⟨.program ⟨214⟩, ⟨29809⟩⟩
def transferEvent : Nat := 66249
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 66243 .summary, .result 66065 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66243 .summary)
      LeftBound66077.bound (LeftBound66077.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22695⟩⟩) (rawTerms := some (Proof.Events258.exact66243RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66077.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66065 .summary)
      LeftBound66060.bound (LeftBound66060.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29808⟩⟩) (rawTerms := some (Proof.Events258.exact66065RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66060.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66077.bound, LeftBound66060.bound]
def bound : CoeffClass := .finite ⟨1292516722839998050304, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66077.bound, LeftBound66060.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66077.actual selector witness, LeftBound66060.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66249

namespace LeftBound66273
def owner : Owner := ⟨.program ⟨214⟩, ⟨12953⟩⟩
def transferEvent : Nat := 66273
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 66271 .coefficient) (.predecessor 1 66272 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66271 .coefficient)
      LeftAuthority3131.bound (LeftAuthority3131.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3132RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3131.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3131.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66272 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3131.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3131.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3131.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound66273

namespace LeftBound66278
def owner : Owner := ⟨.program ⟨214⟩, ⟨7206⟩⟩
def transferEvent : Nat := 66278
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 66276 .coefficient) (.predecessor 1 66277 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66276 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66277 .coefficient)
      LeftBound7473.bound (LeftBound7473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7473.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound7473.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound7473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound7473.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66278

namespace LeftBound66283
def owner : Owner := ⟨.program ⟨214⟩, ⟨12954⟩⟩
def transferEvent : Nat := 66283
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 66281 .coefficient, .predecessor 1 66282 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66281 .coefficient)
      LeftBound66278.bound (LeftBound66278.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66278.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66278.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66282 .coefficient)
      LeftBound66273.bound (LeftBound66273.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66275RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66273.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66273.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66278.bound, LeftBound66273.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66278.bound, LeftBound66273.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66278.actual selector witness, LeftBound66273.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66283

namespace LeftBound66287
def owner : Owner := ⟨.program ⟨214⟩, ⟨12955⟩⟩
def transferEvent : Nat := 66287
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 66285 .coefficient, .predecessor 1 66286 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66285 .coefficient)
      LeftBound66283.bound (LeftBound66283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66283.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66283.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66286 .coefficient)
      LeftBound7465.bound (LeftBound7465.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7466RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7465.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66283.bound, LeftBound7465.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66283.bound, LeftBound7465.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66283.actual selector witness, LeftBound7465.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66287

namespace LeftBound66288
def owner : Owner := ⟨.program ⟨214⟩, ⟨12955⟩⟩
def transferEvent : Nat := 66288
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨102⟩⟩]⟩ [⟨.result 7466 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7466 .coefficient)
      LeftBound7465.bound (LeftBound7465.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨102⟩⟩) (rawTerms := some (Proof.Events029.exact7466RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7465.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound7465.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7465.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound7465.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound66288

namespace LeftBound66293
def owner : Owner := ⟨.program ⟨214⟩, ⟨12956⟩⟩
def transferEvent : Nat := 66293
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 66291 .coefficient) (.predecessor 1 66292 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66291 .coefficient)
      LeftBound66287.bound (LeftBound66287.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66287.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66287.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66292 .coefficient)
      LeftAuthority3134.bound (LeftAuthority3134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3134.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3134.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound66287.bound LeftAuthority3134.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66287.bound, LeftAuthority3134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound66287.actual selector witness) * (LeftAuthority3134.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66293

namespace LeftBound66294
def owner : Owner := ⟨.program ⟨214⟩, ⟨12956⟩⟩
def transferEvent : Nat := 66294
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨10130⟩⟩], []⟩ [⟨.result 3135 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3135 .coefficient)
      LeftAuthority3134.bound (LeftAuthority3134.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨10130⟩⟩) (rawTerms := some (Proof.Events012.exact3135RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3134.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3134.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3134.bound []
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority3134.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound66294

namespace LeftBound66295
def owner : Owner := ⟨.program ⟨214⟩, ⟨12956⟩⟩
def transferEvent : Nat := 66295
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 66290 .summary) (.transfer 66294) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66290 .summary)
      LeftBound66288.bound (LeftBound66288.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12955⟩⟩) (rawTerms := some (Proof.Events258.exact66290RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66288.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 66294)
      LeftBound66294.bound (LeftBound66294.actual selector witness) := by
  exact .transfer (LeftBound66294.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound66288.bound LeftBound66294.bound
def bound : CoeffClass := .finite ⟨43264, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66288.bound, LeftBound66294.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound66288.actual selector witness) * (LeftBound66294.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66295

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
