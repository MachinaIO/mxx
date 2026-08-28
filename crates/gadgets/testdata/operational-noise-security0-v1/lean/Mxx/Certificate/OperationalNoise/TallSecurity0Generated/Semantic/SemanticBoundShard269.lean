import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard268

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound40184
def owner : Owner := ⟨.program ⟨214⟩, ⟨28328⟩⟩
def transferEvent : Nat := 40184
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 40179 .summary) (.transfer 40183) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40179 .summary)
      LeftBound40178.bound (LeftBound40178.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26232⟩⟩) (rawTerms := some (Proof.Events156.exact40179RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40178.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 40183)
      LeftBound40183.bound (LeftBound40183.actual selector witness) := by
  exact .transfer (LeftBound40183.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40178.bound LeftBound40183.bound
def bound : CoeffClass := .finite ⟨1292180534353385750528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40178.bound, LeftBound40183.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40178.actual selector witness) * (LeftBound40183.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40184

namespace LeftBound40195
def owner : Owner := ⟨.program ⟨214⟩, ⟨21698⟩⟩
def transferEvent : Nat := 40195
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 40193 .coefficient) (.value (.predecessor 1 40194 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40193 .coefficient)
      LeftAuthority40191.bound (LeftAuthority40191.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40192RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40191.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40191.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40194 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority40191.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40191.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority40191.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound40195

namespace LeftBound40199
def owner : Owner := ⟨.program ⟨214⟩, ⟨21699⟩⟩
def transferEvent : Nat := 40199
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 40197 .coefficient) (.predecessor 1 40198 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40197 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40198 .coefficient)
      LeftBound40195.bound (LeftBound40195.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40196RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40195.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40195.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound40195.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound40195.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound40195.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40199

namespace LeftBound40200
def owner : Owner := ⟨.program ⟨214⟩, ⟨21699⟩⟩
def transferEvent : Nat := 40200
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21696⟩⟩]⟩ [⟨.result 40192 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40192 .coefficient)
      LeftAuthority40191.bound (LeftAuthority40191.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21696⟩⟩) (rawTerms := some (Proof.Events157.exact40192RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40191.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40191.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority40191.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40191.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority40191.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound40200

namespace LeftBound40201
def owner : Owner := ⟨.program ⟨214⟩, ⟨21699⟩⟩
def transferEvent : Nat := 40201
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 40200) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 40200)
      LeftBound40200.bound (LeftBound40200.actual selector witness) := by
  exact .transfer (LeftBound40200.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound40200.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound40200.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound40200.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40201

namespace LeftBound40296
def owner : Owner := ⟨.program ⟨214⟩, ⟨16187⟩⟩
def transferEvent : Nat := 40296
def frameStart : Nat := 40257
def rule : BoundRule := .identity (.predecessor 0 40295 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40295 .coefficient)
      LeftAuthority40293.bound (LeftAuthority40293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40294RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40293.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40293.derived selector witness)

def rawBound : CoeffClass := LeftAuthority40293.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority40293.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound40296

namespace LeftBound40313
def owner : Owner := ⟨.program ⟨214⟩, ⟨16226⟩⟩
def transferEvent : Nat := 40313
def frameStart : Nat := 40257
def rule : BoundRule := .sum [.predecessor 0 40311 .coefficient, .predecessor 1 40312 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40311 .coefficient)
      LeftBound40296.bound (LeftBound40296.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound40296.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40312 .coefficient)
      LeftAuthority40309.bound (LeftAuthority40309.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority40309.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40296.bound, LeftAuthority40309.bound]
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40296.bound, LeftAuthority40309.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40296.actual selector witness, LeftAuthority40309.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40313

namespace LeftBound40316
def owner : Owner := ⟨.program ⟨214⟩, ⟨16227⟩⟩
def transferEvent : Nat := 40316
def frameStart : Nat := 40257
def rule : BoundRule := .identity (.predecessor 0 40315 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40315 .coefficient)
      LeftBound40313.bound (LeftBound40313.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound40313.derived selector witness)

def rawBound : CoeffClass := LeftBound40313.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40313.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound40313.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound40316

namespace LeftBound40322
def owner : Owner := ⟨.program ⟨214⟩, ⟨16228⟩⟩
def transferEvent : Nat := 40322
def frameStart : Nat := 40257
def rule : BoundRule := .product (.predecessor 0 40320 .coefficient) (.predecessor 1 40321 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40320 .coefficient)
      LeftAuthority40318.bound (LeftAuthority40318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40318.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40321 .coefficient)
      LeftBound40316.bound (LeftBound40316.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40317RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40316.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40316.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority40318.bound LeftBound40316.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40318.bound, LeftBound40316.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority40318.actual selector witness) * (LeftBound40316.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40322

namespace LeftBound40330
def owner : Owner := ⟨.program ⟨214⟩, ⟨16229⟩⟩
def transferEvent : Nat := 40330
def frameStart : Nat := 40257
def rule : BoundRule := .sum [.predecessor 0 40328 .coefficient, .predecessor 1 40329 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40328 .coefficient)
      LeftAuthority40326.bound (LeftAuthority40326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40327RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40326.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40326.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40329 .coefficient)
      LeftBound40322.bound (LeftBound40322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40324RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40322.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40322.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority40326.bound, LeftBound40322.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40326.bound, LeftBound40322.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority40326.actual selector witness, LeftBound40322.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40330

namespace LeftBound40334
def owner : Owner := ⟨.program ⟨214⟩, ⟨28327⟩⟩
def transferEvent : Nat := 40334
def frameStart : Nat := 40257
def rule : BoundRule := .product (.predecessor 0 40332 .coefficient) (.predecessor 1 40333 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40332 .coefficient)
      LeftBound40330.bound (LeftBound40330.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40331RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40330.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40330.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40333 .coefficient)
      LeftAuthority40307.bound (LeftAuthority40307.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40308RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40307.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40307.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40330.bound LeftAuthority40307.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40330.bound, LeftAuthority40307.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40330.actual selector witness) * (LeftAuthority40307.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40334

namespace LeftBound40345
def owner : Owner := ⟨.program ⟨214⟩, ⟨18377⟩⟩
def transferEvent : Nat := 40345
def frameStart : Nat := 40257
def rule : BoundRule := .product (.predecessor 0 40343 .coefficient) (.predecessor 1 40344 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40343 .coefficient)
      LeftAuthority40318.bound (LeftAuthority40318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40318.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40344 .coefficient)
      LeftAuthority40341.bound (LeftAuthority40341.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40342RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40341.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40341.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority40318.bound LeftAuthority40341.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40318.bound, LeftAuthority40341.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority40318.actual selector witness) * (LeftAuthority40341.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40345

namespace LeftBound40353
def owner : Owner := ⟨.program ⟨214⟩, ⟨18378⟩⟩
def transferEvent : Nat := 40353
def frameStart : Nat := 40257
def rule : BoundRule := .sum [.predecessor 0 40351 .coefficient, .predecessor 1 40352 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40351 .coefficient)
      LeftAuthority40349.bound (LeftAuthority40349.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40350RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40349.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40349.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40352 .coefficient)
      LeftBound40345.bound (LeftBound40345.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40347RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40345.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40345.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority40349.bound, LeftBound40345.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40349.bound, LeftBound40345.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority40349.actual selector witness, LeftBound40345.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40353

namespace LeftBound40357
def owner : Owner := ⟨.program ⟨214⟩, ⟨28331⟩⟩
def transferEvent : Nat := 40357
def frameStart : Nat := 40257
def rule : BoundRule := .sum [.predecessor 0 40355 .coefficient, .predecessor 1 40356 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40355 .coefficient)
      LeftBound40353.bound (LeftBound40353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40354RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40353.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40353.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40356 .coefficient)
      LeftBound40334.bound (LeftBound40334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40339RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40334.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40334.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40353.bound, LeftBound40334.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40353.bound, LeftBound40334.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40353.actual selector witness, LeftBound40334.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40357

namespace LeftBound40370
def owner : Owner := ⟨.program ⟨214⟩, ⟨28329⟩⟩
def transferEvent : Nat := 40370
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40368 .coefficient, .predecessor 1 40369 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40368 .coefficient)
      LeftBound40199.bound (LeftBound40199.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40367RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40199.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40199.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40369 .coefficient)
      LeftBound40182.bound (LeftBound40182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40189RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40182.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40182.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40199.bound, LeftBound40182.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40199.bound, LeftBound40182.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40199.actual selector witness, LeftBound40182.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40370

namespace LeftBound40373
def owner : Owner := ⟨.program ⟨214⟩, ⟨28329⟩⟩
def transferEvent : Nat := 40373
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 40367 .summary, .result 40189 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40367 .summary)
      LeftBound40201.bound (LeftBound40201.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21699⟩⟩) (rawTerms := some (Proof.Events157.exact40367RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40201.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40189 .summary)
      LeftBound40184.bound (LeftBound40184.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28328⟩⟩) (rawTerms := some (Proof.Events156.exact40189RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40184.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40201.bound, LeftBound40184.bound]
def bound : CoeffClass := .finite ⟨1292180536164689260544, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40201.bound, LeftBound40184.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40201.actual selector witness, LeftBound40184.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40373

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
