import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard526

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound78159
def owner : Owner := ⟨.program ⟨214⟩, ⟨21038⟩⟩
def transferEvent : Nat := 78159
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 78157 .coefficient) (.value (.predecessor 1 78158 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78157 .coefficient)
      LeftAuthority78155.bound (LeftAuthority78155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78155.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78155.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78158 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority78155.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78155.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority78155.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound78159

namespace LeftBound78163
def owner : Owner := ⟨.program ⟨214⟩, ⟨21039⟩⟩
def transferEvent : Nat := 78163
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 78161 .coefficient) (.predecessor 1 78162 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78161 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78162 .coefficient)
      LeftBound78159.bound (LeftBound78159.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78160RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78159.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78159.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound78159.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound78159.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound78159.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78163

namespace LeftBound78164
def owner : Owner := ⟨.program ⟨214⟩, ⟨21039⟩⟩
def transferEvent : Nat := 78164
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21036⟩⟩]⟩ [⟨.result 78156 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78156 .coefficient)
      LeftAuthority78155.bound (LeftAuthority78155.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21036⟩⟩) (rawTerms := some (Proof.Events305.exact78156RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78155.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78155.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority78155.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78155.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority78155.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound78164

namespace LeftBound78165
def owner : Owner := ⟨.program ⟨214⟩, ⟨21039⟩⟩
def transferEvent : Nat := 78165
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 78164) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 78164)
      LeftBound78164.bound (LeftBound78164.actual selector witness) := by
  exact .transfer (LeftBound78164.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound78164.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound78164.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound78164.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78165

namespace LeftBound78260
def owner : Owner := ⟨.program ⟨214⟩, ⟨15699⟩⟩
def transferEvent : Nat := 78260
def frameStart : Nat := 78221
def rule : BoundRule := .identity (.predecessor 0 78259 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78259 .coefficient)
      LeftAuthority78257.bound (LeftAuthority78257.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78258RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78257.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78257.derived selector witness)

def rawBound : CoeffClass := LeftAuthority78257.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78257.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority78257.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound78260

namespace LeftBound78277
def owner : Owner := ⟨.program ⟨214⟩, ⟨15773⟩⟩
def transferEvent : Nat := 78277
def frameStart : Nat := 78221
def rule : BoundRule := .sum [.predecessor 0 78275 .coefficient, .predecessor 1 78276 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78275 .coefficient)
      LeftBound78260.bound (LeftBound78260.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound78260.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78276 .coefficient)
      LeftAuthority78273.bound (LeftAuthority78273.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority78273.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound78260.bound, LeftAuthority78273.bound]
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78260.bound, LeftAuthority78273.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound78260.actual selector witness, LeftAuthority78273.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78277

namespace LeftBound78280
def owner : Owner := ⟨.program ⟨214⟩, ⟨15774⟩⟩
def transferEvent : Nat := 78280
def frameStart : Nat := 78221
def rule : BoundRule := .identity (.predecessor 0 78279 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78279 .coefficient)
      LeftBound78277.bound (LeftBound78277.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound78277.derived selector witness)

def rawBound : CoeffClass := LeftBound78277.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78277.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound78277.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound78280

namespace LeftBound78286
def owner : Owner := ⟨.program ⟨214⟩, ⟨15775⟩⟩
def transferEvent : Nat := 78286
def frameStart : Nat := 78221
def rule : BoundRule := .product (.predecessor 0 78284 .coefficient) (.predecessor 1 78285 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78284 .coefficient)
      LeftAuthority78282.bound (LeftAuthority78282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78282.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78282.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78285 .coefficient)
      LeftBound78280.bound (LeftBound78280.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78281RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78280.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78280.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority78282.bound LeftBound78280.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78282.bound, LeftBound78280.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority78282.actual selector witness) * (LeftBound78280.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78286

namespace LeftBound78294
def owner : Owner := ⟨.program ⟨214⟩, ⟨15776⟩⟩
def transferEvent : Nat := 78294
def frameStart : Nat := 78221
def rule : BoundRule := .sum [.predecessor 0 78292 .coefficient, .predecessor 1 78293 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78292 .coefficient)
      LeftAuthority78290.bound (LeftAuthority78290.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78291RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78290.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78290.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78293 .coefficient)
      LeftBound78286.bound (LeftBound78286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78288RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78286.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78286.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority78290.bound, LeftBound78286.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78290.bound, LeftBound78286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority78290.actual selector witness, LeftBound78286.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78294

namespace LeftBound78298
def owner : Owner := ⟨.program ⟨214⟩, ⟨27413⟩⟩
def transferEvent : Nat := 78298
def frameStart : Nat := 78221
def rule : BoundRule := .product (.predecessor 0 78296 .coefficient) (.predecessor 1 78297 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78296 .coefficient)
      LeftBound78294.bound (LeftBound78294.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78294.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78294.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78297 .coefficient)
      LeftAuthority78271.bound (LeftAuthority78271.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78272RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78271.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78271.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound78294.bound LeftAuthority78271.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78294.bound, LeftAuthority78271.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound78294.actual selector witness) * (LeftAuthority78271.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78298

namespace LeftBound78309
def owner : Owner := ⟨.program ⟨214⟩, ⟨17436⟩⟩
def transferEvent : Nat := 78309
def frameStart : Nat := 78221
def rule : BoundRule := .product (.predecessor 0 78307 .coefficient) (.predecessor 1 78308 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78307 .coefficient)
      LeftAuthority78282.bound (LeftAuthority78282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78282.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78282.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78308 .coefficient)
      LeftAuthority78305.bound (LeftAuthority78305.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78306RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78305.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78305.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority78282.bound LeftAuthority78305.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78282.bound, LeftAuthority78305.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority78282.actual selector witness) * (LeftAuthority78305.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78309

namespace LeftBound78317
def owner : Owner := ⟨.program ⟨214⟩, ⟨17437⟩⟩
def transferEvent : Nat := 78317
def frameStart : Nat := 78221
def rule : BoundRule := .sum [.predecessor 0 78315 .coefficient, .predecessor 1 78316 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78315 .coefficient)
      LeftAuthority78313.bound (LeftAuthority78313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78313.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78316 .coefficient)
      LeftBound78309.bound (LeftBound78309.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78311RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78309.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78309.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority78313.bound, LeftBound78309.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78313.bound, LeftBound78309.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority78313.actual selector witness, LeftBound78309.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78317

namespace LeftBound78321
def owner : Owner := ⟨.program ⟨214⟩, ⟨27418⟩⟩
def transferEvent : Nat := 78321
def frameStart : Nat := 78221
def rule : BoundRule := .sum [.predecessor 0 78319 .coefficient, .predecessor 1 78320 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78319 .coefficient)
      LeftBound78317.bound (LeftBound78317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78318RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78317.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78317.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78320 .coefficient)
      LeftBound78298.bound (LeftBound78298.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78303RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78298.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78298.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound78317.bound, LeftBound78298.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78317.bound, LeftBound78298.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound78317.actual selector witness, LeftBound78298.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78321

namespace LeftBound78334
def owner : Owner := ⟨.program ⟨214⟩, ⟨27415⟩⟩
def transferEvent : Nat := 78334
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 78332 .coefficient, .predecessor 1 78333 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78332 .coefficient)
      LeftBound78163.bound (LeftBound78163.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78331RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78163.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78163.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78333 .coefficient)
      LeftBound78146.bound (LeftBound78146.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78146.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78146.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound78163.bound, LeftBound78146.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78163.bound, LeftBound78146.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound78163.actual selector witness, LeftBound78146.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78334

namespace LeftBound78337
def owner : Owner := ⟨.program ⟨214⟩, ⟨27415⟩⟩
def transferEvent : Nat := 78337
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 78331 .summary, .result 78153 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78331 .summary)
      LeftBound78165.bound (LeftBound78165.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21039⟩⟩) (rawTerms := some (Proof.Events305.exact78331RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound78165.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78153 .summary)
      LeftBound78148.bound (LeftBound78148.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27414⟩⟩) (rawTerms := some (Proof.Events305.exact78153RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound78148.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound78165.bound, LeftBound78148.bound]
def bound : CoeffClass := .finite ⟨1292001236604524572672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78165.bound, LeftBound78148.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound78165.actual selector witness, LeftBound78148.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78337

namespace LeftBound78341
def owner : Owner := ⟨.program ⟨214⟩, ⟨27416⟩⟩
def transferEvent : Nat := 78341
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 78339 .coefficient) (.predecessor 1 78340 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78339 .coefficient)
      LeftBound78334.bound (LeftBound78334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events306.exact78338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78334.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78334.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78340 .coefficient)
      LeftBound5758.bound (LeftBound5758.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5759RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5758.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5758.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound78334.bound LeftBound5758.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78334.bound, LeftBound5758.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound78334.actual selector witness) * (LeftBound5758.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78341

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
