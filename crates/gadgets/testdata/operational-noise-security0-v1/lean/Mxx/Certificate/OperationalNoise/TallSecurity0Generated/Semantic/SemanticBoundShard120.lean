import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard119

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound19449
def owner : Owner := ⟨.program ⟨214⟩, ⟨21203⟩⟩
def transferEvent : Nat := 19449
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21200⟩⟩]⟩ [⟨.result 19441 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19441 .coefficient)
      LeftAuthority19440.bound (LeftAuthority19440.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21200⟩⟩) (rawTerms := some (Proof.Events075.exact19441RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19440.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority19440.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority19440.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound19449

namespace LeftBound19450
def owner : Owner := ⟨.program ⟨214⟩, ⟨21203⟩⟩
def transferEvent : Nat := 19450
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 19449) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 19449)
      LeftBound19449.bound (LeftBound19449.actual selector witness) := by
  exact .transfer (LeftBound19449.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound19449.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound19449.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound19449.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19450

namespace LeftBound19545
def owner : Owner := ⟨.program ⟨214⟩, ⟨15838⟩⟩
def transferEvent : Nat := 19545
def frameStart : Nat := 19506
def rule : BoundRule := .identity (.predecessor 0 19544 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19544 .coefficient)
      LeftAuthority19542.bound (LeftAuthority19542.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19542.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19542.derived selector witness)

def rawBound : CoeffClass := LeftAuthority19542.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19542.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority19542.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound19545

namespace LeftBound19562
def owner : Owner := ⟨.program ⟨214⟩, ⟨15912⟩⟩
def transferEvent : Nat := 19562
def frameStart : Nat := 19506
def rule : BoundRule := .sum [.predecessor 0 19560 .coefficient, .predecessor 1 19561 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19560 .coefficient)
      LeftBound19545.bound (LeftBound19545.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound19545.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19561 .coefficient)
      LeftAuthority19558.bound (LeftAuthority19558.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority19558.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound19545.bound, LeftAuthority19558.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19545.bound, LeftAuthority19558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound19545.actual selector witness, LeftAuthority19558.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19562

namespace LeftBound19565
def owner : Owner := ⟨.program ⟨214⟩, ⟨15913⟩⟩
def transferEvent : Nat := 19565
def frameStart : Nat := 19506
def rule : BoundRule := .identity (.predecessor 0 19564 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19564 .coefficient)
      LeftBound19562.bound (LeftBound19562.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound19562.derived selector witness)

def rawBound : CoeffClass := LeftBound19562.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound19562.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound19565

namespace LeftBound19571
def owner : Owner := ⟨.program ⟨214⟩, ⟨15914⟩⟩
def transferEvent : Nat := 19571
def frameStart : Nat := 19506
def rule : BoundRule := .product (.predecessor 0 19569 .coefficient) (.predecessor 1 19570 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19569 .coefficient)
      LeftAuthority19567.bound (LeftAuthority19567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19567.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19567.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19570 .coefficient)
      LeftBound19565.bound (LeftBound19565.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19566RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19565.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19565.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority19567.bound LeftBound19565.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19567.bound, LeftBound19565.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority19567.actual selector witness) * (LeftBound19565.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19571

namespace LeftBound19579
def owner : Owner := ⟨.program ⟨214⟩, ⟨15915⟩⟩
def transferEvent : Nat := 19579
def frameStart : Nat := 19506
def rule : BoundRule := .sum [.predecessor 0 19577 .coefficient, .predecessor 1 19578 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19577 .coefficient)
      LeftAuthority19575.bound (LeftAuthority19575.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19575.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19575.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19578 .coefficient)
      LeftBound19571.bound (LeftBound19571.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19571.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19571.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority19575.bound, LeftBound19571.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19575.bound, LeftBound19571.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority19575.actual selector witness, LeftBound19571.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19579

namespace LeftBound19583
def owner : Owner := ⟨.program ⟨214⟩, ⟨27695⟩⟩
def transferEvent : Nat := 19583
def frameStart : Nat := 19506
def rule : BoundRule := .product (.predecessor 0 19581 .coefficient) (.predecessor 1 19582 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19581 .coefficient)
      LeftBound19579.bound (LeftBound19579.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19580RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19579.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19579.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19582 .coefficient)
      LeftAuthority19556.bound (LeftAuthority19556.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19557RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19556.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19556.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound19579.bound LeftAuthority19556.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19579.bound, LeftAuthority19556.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound19579.actual selector witness) * (LeftAuthority19556.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19583

namespace LeftBound19594
def owner : Owner := ⟨.program ⟨214⟩, ⟨17239⟩⟩
def transferEvent : Nat := 19594
def frameStart : Nat := 19506
def rule : BoundRule := .product (.predecessor 0 19592 .coefficient) (.predecessor 1 19593 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19592 .coefficient)
      LeftAuthority19567.bound (LeftAuthority19567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19567.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19567.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19593 .coefficient)
      LeftAuthority19590.bound (LeftAuthority19590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19591RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19590.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19590.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority19567.bound LeftAuthority19590.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19567.bound, LeftAuthority19590.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority19567.actual selector witness) * (LeftAuthority19590.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19594

namespace LeftBound19602
def owner : Owner := ⟨.program ⟨214⟩, ⟨17240⟩⟩
def transferEvent : Nat := 19602
def frameStart : Nat := 19506
def rule : BoundRule := .sum [.predecessor 0 19600 .coefficient, .predecessor 1 19601 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19600 .coefficient)
      LeftAuthority19598.bound (LeftAuthority19598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19599RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19598.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19598.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19601 .coefficient)
      LeftBound19594.bound (LeftBound19594.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19596RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19594.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19594.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority19598.bound, LeftBound19594.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19598.bound, LeftBound19594.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority19598.actual selector witness, LeftBound19594.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19602

namespace LeftBound19606
def owner : Owner := ⟨.program ⟨214⟩, ⟨27700⟩⟩
def transferEvent : Nat := 19606
def frameStart : Nat := 19506
def rule : BoundRule := .sum [.predecessor 0 19604 .coefficient, .predecessor 1 19605 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19604 .coefficient)
      LeftBound19602.bound (LeftBound19602.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19603RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19602.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19602.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19605 .coefficient)
      LeftBound19583.bound (LeftBound19583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19583.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19583.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound19602.bound, LeftBound19583.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19602.bound, LeftBound19583.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound19602.actual selector witness, LeftBound19583.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19606

namespace LeftBound19619
def owner : Owner := ⟨.program ⟨214⟩, ⟨27697⟩⟩
def transferEvent : Nat := 19619
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 19617 .coefficient, .predecessor 1 19618 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19617 .coefficient)
      LeftBound19448.bound (LeftBound19448.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19448.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19448.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19618 .coefficient)
      LeftBound19431.bound (LeftBound19431.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19438RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19431.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19431.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound19448.bound, LeftBound19431.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19448.bound, LeftBound19431.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound19448.actual selector witness, LeftBound19431.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19619

namespace LeftBound19622
def owner : Owner := ⟨.program ⟨214⟩, ⟨27697⟩⟩
def transferEvent : Nat := 19622
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 19616 .summary, .result 19438 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19616 .summary)
      LeftBound19450.bound (LeftBound19450.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21203⟩⟩) (rawTerms := some (Proof.Events076.exact19616RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound19450.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19438 .summary)
      LeftBound19433.bound (LeftBound19433.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27696⟩⟩) (rawTerms := some (Proof.Events075.exact19438RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound19433.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound19450.bound, LeftBound19433.bound]
def bound : CoeffClass := .finite ⟨1292046061494565744640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19450.bound, LeftBound19433.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound19450.actual selector witness, LeftBound19433.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19622

namespace LeftBound19626
def owner : Owner := ⟨.program ⟨214⟩, ⟨27698⟩⟩
def transferEvent : Nat := 19626
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 19624 .coefficient) (.predecessor 1 19625 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19624 .coefficient)
      LeftBound19619.bound (LeftBound19619.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19623RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19619.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19619.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19625 .coefficient)
      LeftBound5738.bound (LeftBound5738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5739RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5738.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5738.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound19619.bound LeftBound5738.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19619.bound, LeftBound5738.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound19619.actual selector witness) * (LeftBound5738.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19626

namespace LeftBound19627
def owner : Owner := ⟨.program ⟨214⟩, ⟨27698⟩⟩
def transferEvent : Nat := 19627
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
end LeftBound19627

namespace LeftBound19628
def owner : Owner := ⟨.program ⟨214⟩, ⟨27698⟩⟩
def transferEvent : Nat := 19628
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 19623 .summary) (.transfer 19627) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19623 .summary)
      LeftBound19622.bound (LeftBound19622.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27697⟩⟩) (rawTerms := some (Proof.Events076.exact19623RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound19622.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 19627)
      LeftBound19627.bound (LeftBound19627.actual selector witness) := by
  exact .transfer (LeftBound19627.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound19622.bound LeftBound19627.bound
def bound : CoeffClass := .finite ⟨4741829718422040195880714240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19622.bound, LeftBound19627.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound19622.actual selector witness) * (LeftBound19627.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19628

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
