import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard080
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard687

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound100068
def owner : Owner := ⟨.program ⟨214⟩, ⟨13535⟩⟩
def transferEvent : Nat := 100068
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 100066 .coefficient, .predecessor 1 100067 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100066 .coefficient)
      LeftBound100064.bound (LeftBound100064.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact100065RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100064.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100064.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100067 .coefficient)
      LeftBound13017.bound (LeftBound13017.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact13018RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13017.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13017.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100064.bound, LeftBound13017.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100064.bound, LeftBound13017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100064.actual selector witness, LeftBound13017.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100068

namespace LeftBound100069
def owner : Owner := ⟨.program ⟨214⟩, ⟨13535⟩⟩
def transferEvent : Nat := 100069
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨107⟩⟩]⟩ [⟨.result 13018 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13018 .coefficient)
      LeftBound13017.bound (LeftBound13017.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨107⟩⟩) (rawTerms := some (Proof.Events050.exact13018RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13017.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13017.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound13017.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound13017.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound100069

namespace LeftBound100074
def owner : Owner := ⟨.program ⟨214⟩, ⟨13536⟩⟩
def transferEvent : Nat := 100074
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 100072 .coefficient) (.predecessor 1 100073 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100072 .coefficient)
      LeftBound100068.bound (LeftBound100068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact100071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100068.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100068.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100073 .coefficient)
      LeftBound13014.bound (LeftBound13014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact13015RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13014.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13014.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound100068.bound LeftBound13014.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100068.bound, LeftBound13014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound100068.actual selector witness) * (LeftBound13014.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100074

namespace LeftBound100075
def owner : Owner := ⟨.program ⟨214⟩, ⟨13536⟩⟩
def transferEvent : Nat := 100075
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩ [⟨.result 13011 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13011 .coefficient)
      LeftAuthority13010.bound (LeftAuthority13010.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7843⟩⟩) (rawTerms := some (Proof.Events050.exact13011RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13010.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13010.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority13010.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13010.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority13010.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound100075

namespace LeftBound100076
def owner : Owner := ⟨.program ⟨214⟩, ⟨13536⟩⟩
def transferEvent : Nat := 100076
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 100071 .summary) (.transfer 100075) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100071 .summary)
      LeftBound100069.bound (LeftBound100069.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13535⟩⟩) (rawTerms := some (Proof.Events390.exact100071RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100069.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 100075)
      LeftBound100075.bound (LeftBound100075.actual selector witness) := by
  exact .transfer (LeftBound100075.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound100069.bound LeftBound100075.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100069.bound, LeftBound100075.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound100069.actual selector witness) * (LeftBound100075.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100076

namespace LeftBound100084
def owner : Owner := ⟨.program ⟨214⟩, ⟨13537⟩⟩
def transferEvent : Nat := 100084
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 100082 .coefficient, .predecessor 1 100083 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100082 .coefficient)
      LeftBound100074.bound (LeftBound100074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact100081RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100074.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100074.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100083 .coefficient)
      LeftBound100046.bound (LeftBound100046.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact100051RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100046.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100046.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100074.bound, LeftBound100046.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100074.bound, LeftBound100046.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100074.actual selector witness, LeftBound100046.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100084

namespace LeftBound100086
def owner : Owner := ⟨.program ⟨214⟩, ⟨13537⟩⟩
def transferEvent : Nat := 100086
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 100081 .summary, .result 100051 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100081 .summary)
      LeftBound100076.bound (LeftBound100076.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13536⟩⟩) (rawTerms := some (Proof.Events390.exact100081RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100076.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100051 .summary)
      LeftBound100048.bound (LeftBound100048.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13532⟩⟩) (rawTerms := some (Proof.Events390.exact100051RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100048.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100076.bound, LeftBound100048.bound]
def bound : CoeffClass := .finite ⟨95428736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100076.bound, LeftBound100048.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100076.actual selector witness, LeftBound100048.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100086

namespace LeftBound100090
def owner : Owner := ⟨.program ⟨214⟩, ⟨25823⟩⟩
def transferEvent : Nat := 100090
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 100088 .coefficient) (.predecessor 1 100089 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100088 .coefficient)
      LeftBound100084.bound (LeftBound100084.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact100087RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100084.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100084.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100089 .coefficient)
      LeftAuthority100022.bound (LeftAuthority100022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact100023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100022.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100022.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound100084.bound LeftAuthority100022.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100084.bound, LeftAuthority100022.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound100084.actual selector witness) * (LeftAuthority100022.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100090

namespace LeftBound100091
def owner : Owner := ⟨.program ⟨214⟩, ⟨25823⟩⟩
def transferEvent : Nat := 100091
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25822⟩⟩]⟩ [⟨.result 100023 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100023 .coefficient)
      LeftAuthority100022.bound (LeftAuthority100022.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25822⟩⟩) (rawTerms := some (Proof.Events390.exact100023RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100022.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100022.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority100022.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100022.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority100022.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound100091

namespace LeftBound100092
def owner : Owner := ⟨.program ⟨214⟩, ⟨25823⟩⟩
def transferEvent : Nat := 100092
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 100087 .summary) (.transfer 100091) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100087 .summary)
      LeftBound100086.bound (LeftBound100086.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13537⟩⟩) (rawTerms := some (Proof.Events390.exact100087RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100086.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 100091)
      LeftBound100091.bound (LeftBound100091.actual selector witness) := by
  exact .transfer (LeftBound100091.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound100086.bound LeftBound100091.bound
def bound : CoeffClass := .finite ⟨350224987979776, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100086.bound, LeftBound100091.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound100086.actual selector witness) * (LeftBound100091.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100092

namespace LeftBound100103
def owner : Owner := ⟨.program ⟨214⟩, ⟨19303⟩⟩
def transferEvent : Nat := 100103
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 100101 .coefficient) (.value (.predecessor 1 100102 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100101 .coefficient)
      LeftAuthority100099.bound (LeftAuthority100099.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100100RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100099.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100099.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100102 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority100099.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100099.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority100099.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound100103

namespace LeftBound100107
def owner : Owner := ⟨.program ⟨214⟩, ⟨19304⟩⟩
def transferEvent : Nat := 100107
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 100105 .coefficient) (.predecessor 1 100106 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100105 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100106 .coefficient)
      LeftBound100103.bound (LeftBound100103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100103.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100103.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound100103.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound100103.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound100103.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100107

namespace LeftBound100108
def owner : Owner := ⟨.program ⟨214⟩, ⟨19304⟩⟩
def transferEvent : Nat := 100108
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19301⟩⟩]⟩ [⟨.result 100100 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100100 .coefficient)
      LeftAuthority100099.bound (LeftAuthority100099.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19301⟩⟩) (rawTerms := some (Proof.Events391.exact100100RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100099.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100099.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority100099.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100099.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority100099.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound100108

namespace LeftBound100109
def owner : Owner := ⟨.program ⟨214⟩, ⟨19304⟩⟩
def transferEvent : Nat := 100109
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 100108) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 100108)
      LeftBound100108.bound (LeftBound100108.actual selector witness) := by
  exact .transfer (LeftBound100108.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound100108.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound100108.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound100108.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100109

namespace LeftBound100164
def owner : Owner := ⟨.program ⟨214⟩, ⟨13530⟩⟩
def transferEvent : Nat := 100164
def frameStart : Nat := 100147
def rule : BoundRule := .product (.predecessor 0 100162 .coefficient) (.predecessor 1 100163 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100162 .coefficient)
      LeftAuthority100160.bound (LeftAuthority100160.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100161RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100160.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100160.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100163 .coefficient)
      LeftAuthority100157.bound (LeftAuthority100157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100158RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100157.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100157.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority100160.bound LeftAuthority100157.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100160.bound, LeftAuthority100157.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority100160.actual selector witness) * (LeftAuthority100157.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100164

namespace LeftBound100168
def owner : Owner := ⟨.program ⟨214⟩, ⟨13531⟩⟩
def transferEvent : Nat := 100168
def frameStart : Nat := 100147
def rule : BoundRule := .identity (.predecessor 0 100167 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100167 .coefficient)
      LeftBound100164.bound (LeftBound100164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100166RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100164.derived selector witness)

def rawBound : CoeffClass := LeftBound100164.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100164.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound100164.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound100168

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
