import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard052
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard053
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard237
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard258

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound38924
def owner : Owner := ⟨.program ⟨214⟩, ⟨28980⟩⟩
def transferEvent : Nat := 38924
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 38922 .coefficient, .predecessor 1 38923 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38922 .coefficient)
      LeftBound38753.bound (LeftBound38753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38753.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38753.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38923 .coefficient)
      LeftBound38736.bound (LeftBound38736.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38743RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38736.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38736.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38753.bound, LeftBound38736.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38753.bound, LeftBound38736.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38753.actual selector witness, LeftBound38736.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38924

namespace LeftBound38927
def owner : Owner := ⟨.program ⟨214⟩, ⟨28980⟩⟩
def transferEvent : Nat := 38927
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 38921 .summary, .result 38743 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38921 .summary)
      LeftBound38755.bound (LeftBound38755.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22131⟩⟩) (rawTerms := some (Proof.Events152.exact38921RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38755.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38743 .summary)
      LeftBound38738.bound (LeftBound38738.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28979⟩⟩) (rawTerms := some (Proof.Events151.exact38743RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38738.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38755.bound, LeftBound38738.bound]
def bound : CoeffClass := .finite ⟨1292315010834812776448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38755.bound, LeftBound38738.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38755.actual selector witness, LeftBound38738.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38927

namespace LeftBound38951
def owner : Owner := ⟨.program ⟨214⟩, ⟨11976⟩⟩
def transferEvent : Nat := 38951
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 38949 .coefficient) (.predecessor 1 38950 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38949 .coefficient)
      LeftAuthority1727.bound (LeftAuthority1727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1727.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1727.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38950 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1727.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1727.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1727.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound38951

namespace LeftBound38956
def owner : Owner := ⟨.program ⟨214⟩, ⟨7316⟩⟩
def transferEvent : Nat := 38956
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 38954 .coefficient) (.predecessor 1 38955 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38954 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38955 .coefficient)
      LeftBound9477.bound (LeftBound9477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9477.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound9477.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound9477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound9477.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38956

namespace LeftBound38961
def owner : Owner := ⟨.program ⟨214⟩, ⟨11977⟩⟩
def transferEvent : Nat := 38961
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 38959 .coefficient, .predecessor 1 38960 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38959 .coefficient)
      LeftBound38956.bound (LeftBound38956.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38958RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38956.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38960 .coefficient)
      LeftBound38951.bound (LeftBound38951.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38953RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38951.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38951.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38956.bound, LeftBound38951.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38956.bound, LeftBound38951.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38956.actual selector witness, LeftBound38951.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38961

namespace LeftBound38965
def owner : Owner := ⟨.program ⟨214⟩, ⟨11978⟩⟩
def transferEvent : Nat := 38965
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 38963 .coefficient, .predecessor 1 38964 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38963 .coefficient)
      LeftBound38961.bound (LeftBound38961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38962RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38961.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38961.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38964 .coefficient)
      LeftBound9469.bound (LeftBound9469.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9470RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9469.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9469.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38961.bound, LeftBound9469.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38961.bound, LeftBound9469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38961.actual selector witness, LeftBound9469.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38965

namespace LeftBound38966
def owner : Owner := ⟨.program ⟨214⟩, ⟨11978⟩⟩
def transferEvent : Nat := 38966
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨98⟩⟩]⟩ [⟨.result 9470 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9470 .coefficient)
      LeftBound9469.bound (LeftBound9469.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨98⟩⟩) (rawTerms := some (Proof.Events036.exact9470RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9469.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9469.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound9469.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound9469.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound38966

namespace LeftBound38971
def owner : Owner := ⟨.program ⟨214⟩, ⟨11979⟩⟩
def transferEvent : Nat := 38971
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 38969 .coefficient) (.predecessor 1 38970 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38969 .coefficient)
      LeftBound38965.bound (LeftBound38965.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38965.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38965.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38970 .coefficient)
      LeftAuthority1730.bound (LeftAuthority1730.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1731RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1730.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1730.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound38965.bound LeftAuthority1730.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38965.bound, LeftAuthority1730.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound38965.actual selector witness) * (LeftAuthority1730.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38971

namespace LeftBound38972
def owner : Owner := ⟨.program ⟨214⟩, ⟨11979⟩⟩
def transferEvent : Nat := 38972
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9725⟩⟩], []⟩ [⟨.result 1731 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 1731 .coefficient)
      LeftAuthority1730.bound (LeftAuthority1730.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9725⟩⟩) (rawTerms := some (Proof.Events006.exact1731RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1730.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1730.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1730.bound []
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1730.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority1730.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound38972

namespace LeftBound38973
def owner : Owner := ⟨.program ⟨214⟩, ⟨11979⟩⟩
def transferEvent : Nat := 38973
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 38968 .summary) (.transfer 38972) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38968 .summary)
      LeftBound38966.bound (LeftBound38966.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11978⟩⟩) (rawTerms := some (Proof.Events152.exact38968RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38966.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 38972)
      LeftBound38972.bound (LeftBound38972.actual selector witness) := by
  exact .transfer (LeftBound38972.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound38966.bound LeftBound38972.bound
def bound : CoeffClass := .finite ⟨29952, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38966.bound, LeftBound38972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound38966.actual selector witness) * (LeftBound38972.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38973

namespace LeftBound38979
def owner : Owner := ⟨.program ⟨214⟩, ⟨9726⟩⟩
def transferEvent : Nat := 38979
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 38977 .coefficient) (.predecessor 1 38978 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38977 .coefficient)
      LeftAuthority1730.bound (LeftAuthority1730.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1731RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1730.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1730.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38978 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1730.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1730.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1730.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound38979

namespace LeftBound38984
def owner : Owner := ⟨.program ⟨214⟩, ⟨7296⟩⟩
def transferEvent : Nat := 38984
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 38982 .coefficient) (.predecessor 1 38983 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38982 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38983 .coefficient)
      LeftBound9518.bound (LeftBound9518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9519RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9518.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound9518.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound9518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound9518.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38984

namespace LeftBound38989
def owner : Owner := ⟨.program ⟨214⟩, ⟨9727⟩⟩
def transferEvent : Nat := 38989
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 38987 .coefficient, .predecessor 1 38988 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38987 .coefficient)
      LeftBound38984.bound (LeftBound38984.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38986RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38984.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38984.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38988 .coefficient)
      LeftBound38979.bound (LeftBound38979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38979.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38979.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38984.bound, LeftBound38979.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38984.bound, LeftBound38979.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38984.actual selector witness, LeftBound38979.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38989

namespace LeftBound38993
def owner : Owner := ⟨.program ⟨214⟩, ⟨9728⟩⟩
def transferEvent : Nat := 38993
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 38991 .coefficient, .predecessor 1 38992 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38991 .coefficient)
      LeftBound38989.bound (LeftBound38989.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38990RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38989.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38989.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38992 .coefficient)
      LeftBound9510.bound (LeftBound9510.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9511RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9510.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9510.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38989.bound, LeftBound9510.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38989.bound, LeftBound9510.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38989.actual selector witness, LeftBound9510.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38993

namespace LeftBound38994
def owner : Owner := ⟨.program ⟨214⟩, ⟨9728⟩⟩
def transferEvent : Nat := 38994
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨78⟩⟩]⟩ [⟨.result 9511 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9511 .coefficient)
      LeftBound9510.bound (LeftBound9510.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨78⟩⟩) (rawTerms := some (Proof.Events037.exact9511RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9510.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9510.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound9510.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9510.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound9510.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound38994

namespace LeftBound38999
def owner : Owner := ⟨.program ⟨214⟩, ⟨9729⟩⟩
def transferEvent : Nat := 38999
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 38997 .coefficient) (.predecessor 1 38998 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 38997 .coefficient)
      LeftBound38993.bound (LeftBound38993.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38996RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38993.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 38998 .coefficient)
      LeftBound9507.bound (LeftBound9507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9507.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9507.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound38993.bound LeftBound9507.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38993.bound, LeftBound9507.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound38993.actual selector witness) * (LeftBound9507.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38999

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
