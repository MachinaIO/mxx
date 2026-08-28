import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard239

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound36328
def owner : Owner := ⟨.program ⟨214⟩, ⟨30163⟩⟩
def transferEvent : Nat := 36328
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36323 .summary) (.transfer 36327) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36323 .summary)
      LeftBound36322.bound (LeftBound36322.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25770⟩⟩) (rawTerms := some (Proof.Events141.exact36323RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36322.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 36327)
      LeftBound36327.bound (LeftBound36327.actual selector witness) := by
  exact .transfer (LeftBound36327.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36322.bound LeftBound36327.bound
def bound : CoeffClass := .finite ⟨1292539133473715126272, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36322.bound, LeftBound36327.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36322.actual selector witness) * (LeftBound36327.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36328

namespace LeftBound36339
def owner : Owner := ⟨.program ⟨214⟩, ⟨22850⟩⟩
def transferEvent : Nat := 36339
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 36337 .coefficient) (.value (.predecessor 1 36338 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36337 .coefficient)
      LeftAuthority36335.bound (LeftAuthority36335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36335.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36335.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36338 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority36335.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36335.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority36335.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound36339

namespace LeftBound36343
def owner : Owner := ⟨.program ⟨214⟩, ⟨22851⟩⟩
def transferEvent : Nat := 36343
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 36341 .coefficient) (.predecessor 1 36342 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36341 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36342 .coefficient)
      LeftBound36339.bound (LeftBound36339.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36340RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36339.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36339.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound36339.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound36339.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound36339.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36343

namespace LeftBound36344
def owner : Owner := ⟨.program ⟨214⟩, ⟨22851⟩⟩
def transferEvent : Nat := 36344
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22848⟩⟩]⟩ [⟨.result 36336 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36336 .coefficient)
      LeftAuthority36335.bound (LeftAuthority36335.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22848⟩⟩) (rawTerms := some (Proof.Events141.exact36336RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36335.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36335.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority36335.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36335.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority36335.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound36344

namespace LeftBound36345
def owner : Owner := ⟨.program ⟨214⟩, ⟨22851⟩⟩
def transferEvent : Nat := 36345
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 36344) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 36344)
      LeftBound36344.bound (LeftBound36344.actual selector witness) := by
  exact .transfer (LeftBound36344.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound36344.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound36344.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound36344.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36345

namespace LeftBound36440
def owner : Owner := ⟨.program ⟨214⟩, ⟨17020⟩⟩
def transferEvent : Nat := 36440
def frameStart : Nat := 36401
def rule : BoundRule := .identity (.predecessor 0 36439 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36439 .coefficient)
      LeftAuthority36437.bound (LeftAuthority36437.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36438RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36437.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36437.derived selector witness)

def rawBound : CoeffClass := LeftAuthority36437.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36437.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority36437.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound36440

namespace LeftBound36457
def owner : Owner := ⟨.program ⟨214⟩, ⟨17059⟩⟩
def transferEvent : Nat := 36457
def frameStart : Nat := 36401
def rule : BoundRule := .sum [.predecessor 0 36455 .coefficient, .predecessor 1 36456 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36455 .coefficient)
      LeftBound36440.bound (LeftBound36440.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound36440.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36456 .coefficient)
      LeftAuthority36453.bound (LeftAuthority36453.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority36453.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36440.bound, LeftAuthority36453.bound]
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36440.bound, LeftAuthority36453.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36440.actual selector witness, LeftAuthority36453.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36457

namespace LeftBound36460
def owner : Owner := ⟨.program ⟨214⟩, ⟨17060⟩⟩
def transferEvent : Nat := 36460
def frameStart : Nat := 36401
def rule : BoundRule := .identity (.predecessor 0 36459 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36459 .coefficient)
      LeftBound36457.bound (LeftBound36457.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound36457.derived selector witness)

def rawBound : CoeffClass := LeftBound36457.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36457.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound36457.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound36460

namespace LeftBound36466
def owner : Owner := ⟨.program ⟨214⟩, ⟨17061⟩⟩
def transferEvent : Nat := 36466
def frameStart : Nat := 36401
def rule : BoundRule := .product (.predecessor 0 36464 .coefficient) (.predecessor 1 36465 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36464 .coefficient)
      LeftAuthority36462.bound (LeftAuthority36462.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36462.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36462.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36465 .coefficient)
      LeftBound36460.bound (LeftBound36460.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36461RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36460.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36460.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority36462.bound LeftBound36460.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36462.bound, LeftBound36460.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority36462.actual selector witness) * (LeftBound36460.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36466

namespace LeftBound36474
def owner : Owner := ⟨.program ⟨214⟩, ⟨17062⟩⟩
def transferEvent : Nat := 36474
def frameStart : Nat := 36401
def rule : BoundRule := .sum [.predecessor 0 36472 .coefficient, .predecessor 1 36473 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36472 .coefficient)
      LeftAuthority36470.bound (LeftAuthority36470.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36471RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36470.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36470.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36473 .coefficient)
      LeftBound36466.bound (LeftBound36466.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36466.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36466.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority36470.bound, LeftBound36466.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36470.bound, LeftBound36466.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority36470.actual selector witness, LeftBound36466.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36474

namespace LeftBound36478
def owner : Owner := ⟨.program ⟨214⟩, ⟨30162⟩⟩
def transferEvent : Nat := 36478
def frameStart : Nat := 36401
def rule : BoundRule := .product (.predecessor 0 36476 .coefficient) (.predecessor 1 36477 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36476 .coefficient)
      LeftBound36474.bound (LeftBound36474.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36474.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36474.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36477 .coefficient)
      LeftAuthority36451.bound (LeftAuthority36451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36452RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36451.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36451.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36474.bound LeftAuthority36451.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36474.bound, LeftAuthority36451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36474.actual selector witness) * (LeftAuthority36451.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36478

namespace LeftBound36489
def owner : Owner := ⟨.program ⟨214⟩, ⟨18177⟩⟩
def transferEvent : Nat := 36489
def frameStart : Nat := 36401
def rule : BoundRule := .product (.predecessor 0 36487 .coefficient) (.predecessor 1 36488 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36487 .coefficient)
      LeftAuthority36462.bound (LeftAuthority36462.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36462.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36462.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36488 .coefficient)
      LeftAuthority36485.bound (LeftAuthority36485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36485.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36485.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority36462.bound LeftAuthority36485.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36462.bound, LeftAuthority36485.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority36462.actual selector witness) * (LeftAuthority36485.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36489

namespace LeftBound36497
def owner : Owner := ⟨.program ⟨214⟩, ⟨18178⟩⟩
def transferEvent : Nat := 36497
def frameStart : Nat := 36401
def rule : BoundRule := .sum [.predecessor 0 36495 .coefficient, .predecessor 1 36496 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36495 .coefficient)
      LeftAuthority36493.bound (LeftAuthority36493.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36493.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36496 .coefficient)
      LeftBound36489.bound (LeftBound36489.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36491RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36489.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36489.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority36493.bound, LeftBound36489.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36493.bound, LeftBound36489.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority36493.actual selector witness, LeftBound36489.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36497

namespace LeftBound36501
def owner : Owner := ⟨.program ⟨214⟩, ⟨30169⟩⟩
def transferEvent : Nat := 36501
def frameStart : Nat := 36401
def rule : BoundRule := .sum [.predecessor 0 36499 .coefficient, .predecessor 1 36500 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36499 .coefficient)
      LeftBound36497.bound (LeftBound36497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36497.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36497.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36500 .coefficient)
      LeftBound36478.bound (LeftBound36478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36478.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36478.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36497.bound, LeftBound36478.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36497.bound, LeftBound36478.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36497.actual selector witness, LeftBound36478.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36501

namespace LeftBound36514
def owner : Owner := ⟨.program ⟨214⟩, ⟨30164⟩⟩
def transferEvent : Nat := 36514
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 36512 .coefficient, .predecessor 1 36513 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36512 .coefficient)
      LeftBound36343.bound (LeftBound36343.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36511RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36343.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36343.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36513 .coefficient)
      LeftBound36326.bound (LeftBound36326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36333RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36326.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36326.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36343.bound, LeftBound36326.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36343.bound, LeftBound36326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36343.actual selector witness, LeftBound36326.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36514

namespace LeftBound36517
def owner : Owner := ⟨.program ⟨214⟩, ⟨30164⟩⟩
def transferEvent : Nat := 36517
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 36511 .summary, .result 36333 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36511 .summary)
      LeftBound36345.bound (LeftBound36345.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22851⟩⟩) (rawTerms := some (Proof.Events142.exact36511RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36345.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36333 .summary)
      LeftBound36328.bound (LeftBound36328.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30163⟩⟩) (rawTerms := some (Proof.Events141.exact36333RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36328.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36345.bound, LeftBound36328.bound]
def bound : CoeffClass := .finite ⟨1292539135285018636288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36345.bound, LeftBound36328.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36345.actual selector witness, LeftBound36328.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36517

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
