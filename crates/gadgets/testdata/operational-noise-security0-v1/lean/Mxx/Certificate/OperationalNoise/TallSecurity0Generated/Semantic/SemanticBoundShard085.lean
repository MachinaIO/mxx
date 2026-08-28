import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard084

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound13540
def owner : Owner := ⟨.program ⟨214⟩, ⟨12205⟩⟩
def transferEvent : Nat := 13540
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨106⟩⟩]⟩ [⟨.result 13519 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13519 .coefficient)
      LeftBound13518.bound (LeftBound13518.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨106⟩⟩) (rawTerms := some (Proof.Events052.exact13519RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13518.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound13518.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound13518.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound13540

namespace LeftBound13545
def owner : Owner := ⟨.program ⟨214⟩, ⟨12206⟩⟩
def transferEvent : Nat := 13545
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 13543 .coefficient) (.predecessor 1 13544 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13543 .coefficient)
      LeftBound13539.bound (LeftBound13539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13542RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13544 .coefficient)
      LeftBound13515.bound (LeftBound13515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13515.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13539.bound LeftBound13515.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13539.bound, LeftBound13515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13539.actual selector witness) * (LeftBound13515.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13545

namespace LeftBound13546
def owner : Owner := ⟨.program ⟨214⟩, ⟨12206⟩⟩
def transferEvent : Nat := 13546
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩ [⟨.result 13512 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13512 .coefficient)
      LeftAuthority13511.bound (LeftAuthority13511.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7840⟩⟩) (rawTerms := some (Proof.Events052.exact13512RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13511.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13511.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority13511.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13511.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority13511.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound13546

namespace LeftBound13547
def owner : Owner := ⟨.program ⟨214⟩, ⟨12206⟩⟩
def transferEvent : Nat := 13547
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 13542 .summary) (.transfer 13546) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13542 .summary)
      LeftBound13540.bound (LeftBound13540.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12205⟩⟩) (rawTerms := some (Proof.Events052.exact13542RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13540.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 13546)
      LeftBound13546.bound (LeftBound13546.actual selector witness) := by
  exact .transfer (LeftBound13546.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13540.bound LeftBound13546.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13540.bound, LeftBound13546.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13540.actual selector witness) * (LeftBound13546.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13547

namespace LeftBound13555
def owner : Owner := ⟨.program ⟨214⟩, ⟨12207⟩⟩
def transferEvent : Nat := 13555
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13553 .coefficient, .predecessor 1 13554 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13553 .coefficient)
      LeftBound13545.bound (LeftBound13545.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13545.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13545.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13554 .coefficient)
      LeftBound13504.bound (LeftBound13504.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13509RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13504.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13504.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13545.bound, LeftBound13504.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13545.bound, LeftBound13504.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13545.actual selector witness, LeftBound13504.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13555

namespace LeftBound13557
def owner : Owner := ⟨.program ⟨214⟩, ⟨12207⟩⟩
def transferEvent : Nat := 13557
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 13552 .summary, .result 13509 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13552 .summary)
      LeftBound13547.bound (LeftBound13547.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12206⟩⟩) (rawTerms := some (Proof.Events052.exact13552RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13547.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13509 .summary)
      LeftBound13506.bound (LeftBound13506.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12202⟩⟩) (rawTerms := some (Proof.Events052.exact13509RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13506.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13547.bound, LeftBound13506.bound]
def bound : CoeffClass := .finite ⟨95425408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13547.bound, LeftBound13506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13547.actual selector witness, LeftBound13506.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13557

namespace LeftBound13561
def owner : Owner := ⟨.program ⟨214⟩, ⟨25317⟩⟩
def transferEvent : Nat := 13561
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 13559 .coefficient) (.predecessor 1 13560 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13559 .coefficient)
      LeftBound13555.bound (LeftBound13555.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13558RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13555.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13555.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13560 .coefficient)
      LeftAuthority13474.bound (LeftAuthority13474.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13474.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13474.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13555.bound LeftAuthority13474.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13555.bound, LeftAuthority13474.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13555.actual selector witness) * (LeftAuthority13474.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13561

namespace LeftBound13562
def owner : Owner := ⟨.program ⟨214⟩, ⟨25317⟩⟩
def transferEvent : Nat := 13562
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25316⟩⟩]⟩ [⟨.result 13475 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13475 .coefficient)
      LeftAuthority13474.bound (LeftAuthority13474.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25316⟩⟩) (rawTerms := some (Proof.Events052.exact13475RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13474.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13474.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority13474.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13474.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority13474.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound13562

namespace LeftBound13563
def owner : Owner := ⟨.program ⟨214⟩, ⟨25317⟩⟩
def transferEvent : Nat := 13563
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 13558 .summary) (.transfer 13562) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13558 .summary)
      LeftBound13557.bound (LeftBound13557.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12207⟩⟩) (rawTerms := some (Proof.Events052.exact13558RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13557.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 13562)
      LeftBound13562.bound (LeftBound13562.actual selector witness) := by
  exact .transfer (LeftBound13562.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13557.bound LeftBound13562.bound
def bound : CoeffClass := .finite ⟨350212774166528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13557.bound, LeftBound13562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13557.actual selector witness) * (LeftBound13562.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13563

namespace LeftBound13574
def owner : Owner := ⟨.program ⟨214⟩, ⟨19258⟩⟩
def transferEvent : Nat := 13574
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 13572 .coefficient) (.value (.predecessor 1 13573 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13572 .coefficient)
      LeftAuthority13570.bound (LeftAuthority13570.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13571RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13570.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13570.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13573 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority13570.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13570.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority13570.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound13574

namespace LeftBound13578
def owner : Owner := ⟨.program ⟨214⟩, ⟨19259⟩⟩
def transferEvent : Nat := 13578
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 13576 .coefficient) (.predecessor 1 13577 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13576 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13577 .coefficient)
      LeftBound13574.bound (LeftBound13574.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13575RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13574.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13574.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound13574.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound13574.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound13574.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13578

namespace LeftBound13579
def owner : Owner := ⟨.program ⟨214⟩, ⟨19259⟩⟩
def transferEvent : Nat := 13579
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19256⟩⟩]⟩ [⟨.result 13571 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13571 .coefficient)
      LeftAuthority13570.bound (LeftAuthority13570.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19256⟩⟩) (rawTerms := some (Proof.Events053.exact13571RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13570.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13570.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority13570.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13570.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority13570.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound13579

namespace LeftBound13580
def owner : Owner := ⟨.program ⟨214⟩, ⟨19259⟩⟩
def transferEvent : Nat := 13580
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 13579) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 13579)
      LeftBound13579.bound (LeftBound13579.actual selector witness) := by
  exact .transfer (LeftBound13579.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound13579.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound13579.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound13579.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13580

namespace LeftBound13659
def owner : Owner := ⟨.program ⟨214⟩, ⟨12200⟩⟩
def transferEvent : Nat := 13659
def frameStart : Nat := 13630
def rule : BoundRule := .product (.predecessor 0 13657 .coefficient) (.predecessor 1 13658 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13657 .coefficient)
      LeftAuthority13655.bound (LeftAuthority13655.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13656RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13655.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13655.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13658 .coefficient)
      LeftAuthority13652.bound (LeftAuthority13652.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13652.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13652.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority13655.bound LeftAuthority13652.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13655.bound, LeftAuthority13652.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority13655.actual selector witness) * (LeftAuthority13652.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13659

namespace LeftBound13663
def owner : Owner := ⟨.program ⟨214⟩, ⟨12201⟩⟩
def transferEvent : Nat := 13663
def frameStart : Nat := 13630
def rule : BoundRule := .identity (.predecessor 0 13662 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13662 .coefficient)
      LeftBound13659.bound (LeftBound13659.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13659.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13659.derived selector witness)

def rawBound : CoeffClass := LeftBound13659.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13659.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound13659.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound13663

namespace LeftBound13680
def owner : Owner := ⟨.program ⟨214⟩, ⟨12286⟩⟩
def transferEvent : Nat := 13680
def frameStart : Nat := 13630
def rule : BoundRule := .sum [.predecessor 0 13678 .coefficient, .predecessor 1 13679 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13678 .coefficient)
      LeftBound13663.bound (LeftBound13663.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound13663.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13679 .coefficient)
      LeftAuthority13676.bound (LeftAuthority13676.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority13676.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13663.bound, LeftAuthority13676.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13663.bound, LeftAuthority13676.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13663.actual selector witness, LeftAuthority13676.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13680

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
