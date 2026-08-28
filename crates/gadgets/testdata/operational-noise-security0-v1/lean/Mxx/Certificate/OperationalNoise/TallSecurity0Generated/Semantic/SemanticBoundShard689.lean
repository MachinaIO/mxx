import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard688

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound100185
def owner : Owner := ⟨.program ⟨214⟩, ⟨13655⟩⟩
def transferEvent : Nat := 100185
def frameStart : Nat := 100147
def rule : BoundRule := .sum [.predecessor 0 100183 .coefficient, .predecessor 1 100184 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100183 .coefficient)
      LeftBound100168.bound (LeftBound100168.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound100168.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100184 .coefficient)
      LeftAuthority100181.bound (LeftAuthority100181.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority100181.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100168.bound, LeftAuthority100181.bound]
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100168.bound, LeftAuthority100181.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100168.actual selector witness, LeftAuthority100181.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100185

namespace LeftBound100188
def owner : Owner := ⟨.program ⟨214⟩, ⟨13656⟩⟩
def transferEvent : Nat := 100188
def frameStart : Nat := 100147
def rule : BoundRule := .identity (.predecessor 0 100187 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100187 .coefficient)
      LeftBound100185.bound (LeftBound100185.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound100185.derived selector witness)

def rawBound : CoeffClass := LeftBound100185.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100185.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound100185.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound100188

namespace LeftBound100194
def owner : Owner := ⟨.program ⟨214⟩, ⟨13657⟩⟩
def transferEvent : Nat := 100194
def frameStart : Nat := 100147
def rule : BoundRule := .product (.predecessor 0 100192 .coefficient) (.predecessor 1 100193 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100192 .coefficient)
      LeftAuthority100190.bound (LeftAuthority100190.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100190.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100190.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100193 .coefficient)
      LeftBound100188.bound (LeftBound100188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100189RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100188.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100188.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority100190.bound LeftBound100188.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100190.bound, LeftBound100188.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority100190.actual selector witness) * (LeftBound100188.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100194

namespace LeftBound100210
def owner : Owner := ⟨.program ⟨214⟩, ⟨7844⟩⟩
def transferEvent : Nat := 100210
def frameStart : Nat := 100147
def rule : BoundRule := .scale (.predecessor 0 100208 .coefficient) (.value (.predecessor 1 100209 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100208 .coefficient)
      LeftAuthority100206.bound (LeftAuthority100206.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100207RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100206.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100206.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100209 .coefficient)
      LeftAuthority100197.bound (LeftAuthority100197.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority100197.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority100206.bound LeftAuthority100197.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100206.bound, LeftAuthority100197.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority100206.actual selector witness) * (LeftAuthority100197.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound100210

namespace LeftBound100213
def owner : Owner := ⟨.program ⟨214⟩, ⟨6793⟩⟩
def transferEvent : Nat := 100213
def frameStart : Nat := 100147
def rule : BoundRule := .identity (.predecessor 0 100212 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100212 .coefficient)
      LeftAuthority100200.bound (LeftAuthority100200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100200.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100200.derived selector witness)

def rawBound : CoeffClass := LeftAuthority100200.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100200.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority100200.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound100213

namespace LeftBound100217
def owner : Owner := ⟨.program ⟨214⟩, ⟨7845⟩⟩
def transferEvent : Nat := 100217
def frameStart : Nat := 100147
def rule : BoundRule := .product (.predecessor 0 100215 .coefficient) (.predecessor 1 100216 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100215 .coefficient)
      LeftBound100213.bound (LeftBound100213.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100214RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100213.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100213.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100216 .coefficient)
      LeftBound100210.bound (LeftBound100210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100210.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100210.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound100213.bound LeftBound100210.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100213.bound, LeftBound100210.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound100213.actual selector witness) * (LeftBound100210.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100217

namespace LeftBound100222
def owner : Owner := ⟨.program ⟨214⟩, ⟨13658⟩⟩
def transferEvent : Nat := 100222
def frameStart : Nat := 100147
def rule : BoundRule := .sum [.predecessor 0 100220 .coefficient, .predecessor 1 100221 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100220 .coefficient)
      LeftBound100217.bound (LeftBound100217.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100219RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100217.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100217.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100221 .coefficient)
      LeftBound100194.bound (LeftBound100194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100196RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100194.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100194.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100217.bound, LeftBound100194.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100217.bound, LeftBound100194.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100217.actual selector witness, LeftBound100194.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100222

namespace LeftBound100226
def owner : Owner := ⟨.program ⟨214⟩, ⟨25825⟩⟩
def transferEvent : Nat := 100226
def frameStart : Nat := 100147
def rule : BoundRule := .product (.predecessor 0 100224 .coefficient) (.predecessor 1 100225 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100224 .coefficient)
      LeftBound100222.bound (LeftBound100222.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100222.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100222.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100225 .coefficient)
      LeftAuthority100179.bound (LeftAuthority100179.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100180RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100179.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100179.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound100222.bound LeftAuthority100179.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100222.bound, LeftAuthority100179.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound100222.actual selector witness) * (LeftAuthority100179.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100226

namespace LeftBound100237
def owner : Owner := ⟨.program ⟨214⟩, ⟨15575⟩⟩
def transferEvent : Nat := 100237
def frameStart : Nat := 100147
def rule : BoundRule := .product (.predecessor 0 100235 .coefficient) (.predecessor 1 100236 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100235 .coefficient)
      LeftAuthority100190.bound (LeftAuthority100190.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100190.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100190.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100236 .coefficient)
      LeftAuthority100233.bound (LeftAuthority100233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100234RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100233.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100233.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority100190.bound LeftAuthority100233.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100190.bound, LeftAuthority100233.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority100190.actual selector witness) * (LeftAuthority100233.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100237

namespace LeftBound100245
def owner : Owner := ⟨.program ⟨214⟩, ⟨15576⟩⟩
def transferEvent : Nat := 100245
def frameStart : Nat := 100147
def rule : BoundRule := .sum [.predecessor 0 100243 .coefficient, .predecessor 1 100244 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100243 .coefficient)
      LeftAuthority100241.bound (LeftAuthority100241.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100242RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100241.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100241.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100244 .coefficient)
      LeftBound100237.bound (LeftBound100237.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100239RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100237.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100237.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority100241.bound, LeftBound100237.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100241.bound, LeftBound100237.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority100241.actual selector witness, LeftBound100237.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100245

namespace LeftBound100249
def owner : Owner := ⟨.program ⟨214⟩, ⟨25826⟩⟩
def transferEvent : Nat := 100249
def frameStart : Nat := 100147
def rule : BoundRule := .sum [.predecessor 0 100247 .coefficient, .predecessor 1 100248 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100247 .coefficient)
      LeftBound100245.bound (LeftBound100245.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100246RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100245.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100245.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100248 .coefficient)
      LeftBound100226.bound (LeftBound100226.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100231RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100226.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100226.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100245.bound, LeftBound100226.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100245.bound, LeftBound100226.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100245.actual selector witness, LeftBound100226.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100249

namespace LeftBound100262
def owner : Owner := ⟨.program ⟨214⟩, ⟨25824⟩⟩
def transferEvent : Nat := 100262
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 100260 .coefficient, .predecessor 1 100261 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100260 .coefficient)
      LeftBound100107.bound (LeftBound100107.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100259RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100107.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100107.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100261 .coefficient)
      LeftBound100090.bound (LeftBound100090.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100097RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100090.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100090.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100107.bound, LeftBound100090.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100107.bound, LeftBound100090.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100107.actual selector witness, LeftBound100090.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100262

namespace LeftBound100265
def owner : Owner := ⟨.program ⟨214⟩, ⟨25824⟩⟩
def transferEvent : Nat := 100265
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 100259 .summary, .result 100097 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100259 .summary)
      LeftBound100109.bound (LeftBound100109.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19304⟩⟩) (rawTerms := some (Proof.Events391.exact100259RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100109.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100097 .summary)
      LeftBound100092.bound (LeftBound100092.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25823⟩⟩) (rawTerms := some (Proof.Events391.exact100097RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100092.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100109.bound, LeftBound100092.bound]
def bound : CoeffClass := .finite ⟨352036291489792, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100109.bound, LeftBound100092.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100109.actual selector witness, LeftBound100092.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100265

namespace LeftBound100269
def owner : Owner := ⟨.program ⟨214⟩, ⟨27182⟩⟩
def transferEvent : Nat := 100269
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 100267 .coefficient) (.predecessor 1 100268 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100267 .coefficient)
      LeftBound100262.bound (LeftBound100262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100266RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100262.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100262.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100268 .coefficient)
      LeftAuthority100012.bound (LeftAuthority100012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact100013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100012.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100012.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound100262.bound LeftAuthority100012.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100262.bound, LeftAuthority100012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound100262.actual selector witness) * (LeftAuthority100012.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100269

namespace LeftBound100270
def owner : Owner := ⟨.program ⟨214⟩, ⟨27182⟩⟩
def transferEvent : Nat := 100270
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27180⟩⟩]⟩ [⟨.result 100013 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100013 .coefficient)
      LeftAuthority100012.bound (LeftAuthority100012.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27180⟩⟩) (rawTerms := some (Proof.Events390.exact100013RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100012.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100012.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority100012.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority100012.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound100270

namespace LeftBound100271
def owner : Owner := ⟨.program ⟨214⟩, ⟨27182⟩⟩
def transferEvent : Nat := 100271
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 100266 .summary) (.transfer 100270) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100266 .summary)
      LeftBound100265.bound (LeftBound100265.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25824⟩⟩) (rawTerms := some (Proof.Events391.exact100266RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100265.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 100270)
      LeftBound100270.bound (LeftBound100270.actual selector witness) := by
  exact .transfer (LeftBound100270.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound100265.bound LeftBound100270.bound
def bound : CoeffClass := .finite ⟨1291978822348200476672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100265.bound, LeftBound100270.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound100265.actual selector witness) * (LeftBound100270.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100271

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
