import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard036
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard237
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard243

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound36939
def owner : Owner := ⟨.program ⟨214⟩, ⟨16975⟩⟩
def transferEvent : Nat := 36939
def frameStart : Nat := 36883
def rule : BoundRule := .sum [.predecessor 0 36937 .coefficient, .predecessor 1 36938 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36937 .coefficient)
      LeftBound36922.bound (LeftBound36922.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound36922.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36938 .coefficient)
      LeftAuthority36935.bound (LeftAuthority36935.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority36935.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36922.bound, LeftAuthority36935.bound]
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36922.bound, LeftAuthority36935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36922.actual selector witness, LeftAuthority36935.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36939

namespace LeftBound36942
def owner : Owner := ⟨.program ⟨214⟩, ⟨16976⟩⟩
def transferEvent : Nat := 36942
def frameStart : Nat := 36883
def rule : BoundRule := .identity (.predecessor 0 36941 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36941 .coefficient)
      LeftBound36939.bound (LeftBound36939.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound36939.derived selector witness)

def rawBound : CoeffClass := LeftBound36939.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36939.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound36939.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound36942

namespace LeftBound36948
def owner : Owner := ⟨.program ⟨214⟩, ⟨16977⟩⟩
def transferEvent : Nat := 36948
def frameStart : Nat := 36883
def rule : BoundRule := .product (.predecessor 0 36946 .coefficient) (.predecessor 1 36947 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36946 .coefficient)
      LeftAuthority36944.bound (LeftAuthority36944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36944.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36947 .coefficient)
      LeftBound36942.bound (LeftBound36942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36943RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36942.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36942.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority36944.bound LeftBound36942.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36944.bound, LeftBound36942.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority36944.actual selector witness) * (LeftBound36942.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36948

namespace LeftBound36956
def owner : Owner := ⟨.program ⟨214⟩, ⟨16978⟩⟩
def transferEvent : Nat := 36956
def frameStart : Nat := 36883
def rule : BoundRule := .sum [.predecessor 0 36954 .coefficient, .predecessor 1 36955 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36954 .coefficient)
      LeftAuthority36952.bound (LeftAuthority36952.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36953RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36952.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36952.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36955 .coefficient)
      LeftBound36948.bound (LeftBound36948.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36950RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36948.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36948.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority36952.bound, LeftBound36948.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36952.bound, LeftBound36948.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority36952.actual selector witness, LeftBound36948.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36956

namespace LeftBound36960
def owner : Owner := ⟨.program ⟨214⟩, ⟨29846⟩⟩
def transferEvent : Nat := 36960
def frameStart : Nat := 36883
def rule : BoundRule := .product (.predecessor 0 36958 .coefficient) (.predecessor 1 36959 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36958 .coefficient)
      LeftBound36956.bound (LeftBound36956.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36957RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36956.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36959 .coefficient)
      LeftAuthority36933.bound (LeftAuthority36933.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36934RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36933.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36933.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36956.bound LeftAuthority36933.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36956.bound, LeftAuthority36933.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36956.actual selector witness) * (LeftAuthority36933.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36960

namespace LeftBound36971
def owner : Owner := ⟨.program ⟨214⟩, ⟨17092⟩⟩
def transferEvent : Nat := 36971
def frameStart : Nat := 36883
def rule : BoundRule := .product (.predecessor 0 36969 .coefficient) (.predecessor 1 36970 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36969 .coefficient)
      LeftAuthority36944.bound (LeftAuthority36944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36944.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36970 .coefficient)
      LeftAuthority36967.bound (LeftAuthority36967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36967.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36967.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority36944.bound LeftAuthority36967.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36944.bound, LeftAuthority36967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority36944.actual selector witness) * (LeftAuthority36967.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36971

namespace LeftBound36979
def owner : Owner := ⟨.program ⟨214⟩, ⟨17093⟩⟩
def transferEvent : Nat := 36979
def frameStart : Nat := 36883
def rule : BoundRule := .sum [.predecessor 0 36977 .coefficient, .predecessor 1 36978 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36977 .coefficient)
      LeftAuthority36975.bound (LeftAuthority36975.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36975.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36975.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36978 .coefficient)
      LeftBound36971.bound (LeftBound36971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36971.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36971.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority36975.bound, LeftBound36971.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36975.bound, LeftBound36971.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority36975.actual selector witness, LeftBound36971.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36979

namespace LeftBound36983
def owner : Owner := ⟨.program ⟨214⟩, ⟨29850⟩⟩
def transferEvent : Nat := 36983
def frameStart : Nat := 36883
def rule : BoundRule := .sum [.predecessor 0 36981 .coefficient, .predecessor 1 36982 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36981 .coefficient)
      LeftBound36979.bound (LeftBound36979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36979.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36982 .coefficient)
      LeftBound36960.bound (LeftBound36960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36965RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36960.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36979.bound, LeftBound36960.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36979.bound, LeftBound36960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36979.actual selector witness, LeftBound36960.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36983

namespace LeftBound36996
def owner : Owner := ⟨.program ⟨214⟩, ⟨29848⟩⟩
def transferEvent : Nat := 36996
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 36994 .coefficient, .predecessor 1 36995 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36994 .coefficient)
      LeftBound36825.bound (LeftBound36825.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36993RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36825.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36825.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36995 .coefficient)
      LeftBound36808.bound (LeftBound36808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36808.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36808.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36825.bound, LeftBound36808.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36825.bound, LeftBound36808.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36825.actual selector witness, LeftBound36808.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36996

namespace LeftBound36999
def owner : Owner := ⟨.program ⟨214⟩, ⟨29848⟩⟩
def transferEvent : Nat := 36999
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 36993 .summary, .result 36815 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36993 .summary)
      LeftBound36827.bound (LeftBound36827.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22707⟩⟩) (rawTerms := some (Proof.Events144.exact36993RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36827.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36815 .summary)
      LeftBound36810.bound (LeftBound36810.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29847⟩⟩) (rawTerms := some (Proof.Events143.exact36815RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36810.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36827.bound, LeftBound36810.bound]
def bound : CoeffClass := .finite ⟨1292516722839998050304, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36827.bound, LeftBound36810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36827.actual selector witness, LeftBound36810.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36999

namespace LeftBound37023
def owner : Owner := ⟨.program ⟨214⟩, ⟨12977⟩⟩
def transferEvent : Nat := 37023
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 37021 .coefficient) (.predecessor 1 37022 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37021 .coefficient)
      LeftAuthority1635.bound (LeftAuthority1635.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1635.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1635.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37022 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1635.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1635.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1635.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound37023

namespace LeftBound37028
def owner : Owner := ⟨.program ⟨214⟩, ⟨7320⟩⟩
def transferEvent : Nat := 37028
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 37026 .coefficient) (.predecessor 1 37027 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37026 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37027 .coefficient)
      LeftBound7473.bound (LeftBound7473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7473.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound7473.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound7473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound7473.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37028

namespace LeftBound37033
def owner : Owner := ⟨.program ⟨214⟩, ⟨12978⟩⟩
def transferEvent : Nat := 37033
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 37031 .coefficient, .predecessor 1 37032 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37031 .coefficient)
      LeftBound37028.bound (LeftBound37028.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact37030RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37028.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37028.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37032 .coefficient)
      LeftBound37023.bound (LeftBound37023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact37025RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37023.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37023.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37028.bound, LeftBound37023.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37028.bound, LeftBound37023.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound37028.actual selector witness, LeftBound37023.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37033

namespace LeftBound37037
def owner : Owner := ⟨.program ⟨214⟩, ⟨12979⟩⟩
def transferEvent : Nat := 37037
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 37035 .coefficient, .predecessor 1 37036 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37035 .coefficient)
      LeftBound37033.bound (LeftBound37033.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact37034RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37033.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37033.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37036 .coefficient)
      LeftBound7465.bound (LeftBound7465.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7466RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7465.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37033.bound, LeftBound7465.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37033.bound, LeftBound7465.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound37033.actual selector witness, LeftBound7465.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37037

namespace LeftBound37038
def owner : Owner := ⟨.program ⟨214⟩, ⟨12979⟩⟩
def transferEvent : Nat := 37038
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
end LeftBound37038

namespace LeftBound37043
def owner : Owner := ⟨.program ⟨214⟩, ⟨12980⟩⟩
def transferEvent : Nat := 37043
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 37041 .coefficient) (.predecessor 1 37042 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37041 .coefficient)
      LeftBound37037.bound (LeftBound37037.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact37040RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37037.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37037.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37042 .coefficient)
      LeftAuthority1638.bound (LeftAuthority1638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1639RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1638.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1638.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound37037.bound LeftAuthority1638.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37037.bound, LeftAuthority1638.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound37037.actual selector witness) * (LeftAuthority1638.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37043

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
