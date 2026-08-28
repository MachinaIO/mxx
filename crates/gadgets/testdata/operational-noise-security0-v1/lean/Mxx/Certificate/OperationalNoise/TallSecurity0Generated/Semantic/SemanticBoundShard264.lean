import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard263

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound39499
def owner : Owner := ⟨.program ⟨214⟩, ⟨25153⟩⟩
def transferEvent : Nat := 39499
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 39494 .summary) (.transfer 39498) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39494 .summary)
      LeftBound39493.bound (LeftBound39493.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11784⟩⟩) (rawTerms := some (Proof.Events154.exact39494RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 39498)
      LeftBound39498.bound (LeftBound39498.actual selector witness) := by
  exact .transfer (LeftBound39498.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound39493.bound LeftBound39498.bound
def bound : CoeffClass := .finite ⟨350286057046016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39493.bound, LeftBound39498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound39493.actual selector witness) * (LeftBound39498.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39499

namespace LeftBound39510
def owner : Owner := ⟨.program ⟨214⟩, ⟨19754⟩⟩
def transferEvent : Nat := 39510
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 39508 .coefficient) (.value (.predecessor 1 39509 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39508 .coefficient)
      LeftAuthority39506.bound (LeftAuthority39506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39507RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39506.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39506.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39509 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority39506.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39506.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority39506.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound39510

namespace LeftBound39514
def owner : Owner := ⟨.program ⟨214⟩, ⟨19755⟩⟩
def transferEvent : Nat := 39514
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 39512 .coefficient) (.predecessor 1 39513 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39512 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39513 .coefficient)
      LeftBound39510.bound (LeftBound39510.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39511RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39510.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39510.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound39510.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound39510.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound39510.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39514

namespace LeftBound39515
def owner : Owner := ⟨.program ⟨214⟩, ⟨19755⟩⟩
def transferEvent : Nat := 39515
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19752⟩⟩]⟩ [⟨.result 39507 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39507 .coefficient)
      LeftAuthority39506.bound (LeftAuthority39506.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19752⟩⟩) (rawTerms := some (Proof.Events154.exact39507RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39506.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39506.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority39506.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority39506.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound39515

namespace LeftBound39516
def owner : Owner := ⟨.program ⟨214⟩, ⟨19755⟩⟩
def transferEvent : Nat := 39516
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 39515) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 39515)
      LeftBound39515.bound (LeftBound39515.actual selector witness) := by
  exact .transfer (LeftBound39515.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound39515.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound39515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound39515.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39516

namespace LeftBound39595
def owner : Owner := ⟨.program ⟨214⟩, ⟨11778⟩⟩
def transferEvent : Nat := 39595
def frameStart : Nat := 39566
def rule : BoundRule := .product (.predecessor 0 39593 .coefficient) (.predecessor 1 39594 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39593 .coefficient)
      LeftAuthority39591.bound (LeftAuthority39591.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39591.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39591.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39594 .coefficient)
      LeftAuthority39588.bound (LeftAuthority39588.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39588.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39588.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority39591.bound LeftAuthority39588.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39591.bound, LeftAuthority39588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority39591.actual selector witness) * (LeftAuthority39588.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39595

namespace LeftBound39599
def owner : Owner := ⟨.program ⟨214⟩, ⟨11779⟩⟩
def transferEvent : Nat := 39599
def frameStart : Nat := 39566
def rule : BoundRule := .identity (.predecessor 0 39598 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39598 .coefficient)
      LeftBound39595.bound (LeftBound39595.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39597RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39595.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39595.derived selector witness)

def rawBound : CoeffClass := LeftBound39595.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39595.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound39595.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound39599

namespace LeftBound39616
def owner : Owner := ⟨.program ⟨214⟩, ⟨11865⟩⟩
def transferEvent : Nat := 39616
def frameStart : Nat := 39566
def rule : BoundRule := .sum [.predecessor 0 39614 .coefficient, .predecessor 1 39615 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39614 .coefficient)
      LeftBound39599.bound (LeftBound39599.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound39599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39615 .coefficient)
      LeftAuthority39612.bound (LeftAuthority39612.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority39612.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39599.bound, LeftAuthority39612.bound]
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39599.bound, LeftAuthority39612.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39599.actual selector witness, LeftAuthority39612.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39616

namespace LeftBound39619
def owner : Owner := ⟨.program ⟨214⟩, ⟨11866⟩⟩
def transferEvent : Nat := 39619
def frameStart : Nat := 39566
def rule : BoundRule := .identity (.predecessor 0 39618 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39618 .coefficient)
      LeftBound39616.bound (LeftBound39616.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound39616.derived selector witness)

def rawBound : CoeffClass := LeftBound39616.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39616.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound39616.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound39619

namespace LeftBound39625
def owner : Owner := ⟨.program ⟨214⟩, ⟨11867⟩⟩
def transferEvent : Nat := 39625
def frameStart : Nat := 39566
def rule : BoundRule := .product (.predecessor 0 39623 .coefficient) (.predecessor 1 39624 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39623 .coefficient)
      LeftAuthority39621.bound (LeftAuthority39621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39622RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39621.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39621.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39624 .coefficient)
      LeftBound39619.bound (LeftBound39619.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39619.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39619.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority39621.bound LeftBound39619.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39621.bound, LeftBound39619.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority39621.actual selector witness) * (LeftBound39619.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39625

namespace LeftBound39641
def owner : Owner := ⟨.program ⟨214⟩, ⟨7862⟩⟩
def transferEvent : Nat := 39641
def frameStart : Nat := 39566
def rule : BoundRule := .scale (.predecessor 0 39639 .coefficient) (.value (.predecessor 1 39640 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39639 .coefficient)
      LeftAuthority39637.bound (LeftAuthority39637.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39638RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39637.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39637.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39640 .coefficient)
      LeftAuthority39628.bound (LeftAuthority39628.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority39628.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority39637.bound LeftAuthority39628.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39637.bound, LeftAuthority39628.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority39637.actual selector witness) * (LeftAuthority39628.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound39641

namespace LeftBound39644
def owner : Owner := ⟨.program ⟨214⟩, ⟨6763⟩⟩
def transferEvent : Nat := 39644
def frameStart : Nat := 39566
def rule : BoundRule := .identity (.predecessor 0 39643 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39643 .coefficient)
      LeftAuthority39631.bound (LeftAuthority39631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39632RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39631.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39631.derived selector witness)

def rawBound : CoeffClass := LeftAuthority39631.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority39631.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound39644

namespace LeftBound39648
def owner : Owner := ⟨.program ⟨214⟩, ⟨7863⟩⟩
def transferEvent : Nat := 39648
def frameStart : Nat := 39566
def rule : BoundRule := .product (.predecessor 0 39646 .coefficient) (.predecessor 1 39647 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39646 .coefficient)
      LeftBound39644.bound (LeftBound39644.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39645RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39644.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39644.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39647 .coefficient)
      LeftBound39641.bound (LeftBound39641.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39642RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39641.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39641.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound39644.bound LeftBound39641.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39644.bound, LeftBound39641.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound39644.actual selector witness) * (LeftBound39641.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39648

namespace LeftBound39653
def owner : Owner := ⟨.program ⟨214⟩, ⟨11868⟩⟩
def transferEvent : Nat := 39653
def frameStart : Nat := 39566
def rule : BoundRule := .sum [.predecessor 0 39651 .coefficient, .predecessor 1 39652 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39651 .coefficient)
      LeftBound39648.bound (LeftBound39648.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39650RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39648.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39648.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39652 .coefficient)
      LeftBound39625.bound (LeftBound39625.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39627RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39625.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39625.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39648.bound, LeftBound39625.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39648.bound, LeftBound39625.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39648.actual selector witness, LeftBound39625.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39653

namespace LeftBound39657
def owner : Owner := ⟨.program ⟨214⟩, ⟨25155⟩⟩
def transferEvent : Nat := 39657
def frameStart : Nat := 39566
def rule : BoundRule := .product (.predecessor 0 39655 .coefficient) (.predecessor 1 39656 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39655 .coefficient)
      LeftBound39653.bound (LeftBound39653.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39654RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39653.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39653.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39656 .coefficient)
      LeftAuthority39610.bound (LeftAuthority39610.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39610.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39610.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound39653.bound LeftAuthority39610.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39653.bound, LeftAuthority39610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound39653.actual selector witness) * (LeftAuthority39610.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39657

namespace LeftBound39668
def owner : Owner := ⟨.program ⟨214⟩, ⟨16272⟩⟩
def transferEvent : Nat := 39668
def frameStart : Nat := 39566
def rule : BoundRule := .product (.predecessor 0 39666 .coefficient) (.predecessor 1 39667 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39666 .coefficient)
      LeftAuthority39621.bound (LeftAuthority39621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39622RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39621.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39621.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39667 .coefficient)
      LeftAuthority39664.bound (LeftAuthority39664.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39664.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39664.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority39621.bound LeftAuthority39664.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39621.bound, LeftAuthority39664.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority39621.actual selector witness) * (LeftAuthority39664.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39668

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
