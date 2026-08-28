import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard112
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard113
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard114
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard115
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard116
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard117
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard118
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard119
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard120
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard122
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard129

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound20963
def owner : Owner := ⟨.program ⟨214⟩, ⟨27482⟩⟩
def transferEvent : Nat := 20963
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 20961 .coefficient, .predecessor 1 20962 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20961 .coefficient)
      LeftBound20958.bound (LeftBound20958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20960RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20958.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20958.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20962 .coefficient)
      LeftBound19838.bound (LeftBound19838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events077.exact19845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19838.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19838.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20958.bound, LeftBound19838.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20958.bound, LeftBound19838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20958.actual selector witness, LeftBound19838.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20963

namespace LeftBound20964
def owner : Owner := ⟨.program ⟨214⟩, ⟨27482⟩⟩
def transferEvent : Nat := 20964
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 20960 .summary, .result 19845 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20960 .summary)
      LeftBound20959.bound (LeftBound20959.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27265⟩⟩) (rawTerms := some (Proof.Events081.exact20960RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20959.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19845 .summary)
      LeftBound19840.bound (LeftBound19840.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27481⟩⟩) (rawTerms := some (Proof.Events077.exact19845RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound19840.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20959.bound, LeftBound19840.bound]
def bound : CoeffClass := .finite ⟨28448551816593413384009941044, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20959.bound, LeftBound19840.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20959.actual selector witness, LeftBound19840.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20964

namespace LeftBound20968
def owner : Owner := ⟨.program ⟨214⟩, ⟨27699⟩⟩
def transferEvent : Nat := 20968
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 20966 .coefficient, .predecessor 1 20967 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20966 .coefficient)
      LeftBound20963.bound (LeftBound20963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20965RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20963.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20963.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20967 .coefficient)
      LeftBound19626.bound (LeftBound19626.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19633RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19626.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19626.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20963.bound, LeftBound19626.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20963.bound, LeftBound19626.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20963.actual selector witness, LeftBound19626.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20968

namespace LeftBound20969
def owner : Owner := ⟨.program ⟨214⟩, ⟨27699⟩⟩
def transferEvent : Nat := 20969
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 20965 .summary, .result 19633 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20965 .summary)
      LeftBound20964.bound (LeftBound20964.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27482⟩⟩) (rawTerms := some (Proof.Events081.exact20965RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20964.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19633 .summary)
      LeftBound19628.bound (LeftBound19628.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27698⟩⟩) (rawTerms := some (Proof.Events076.exact19633RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound19628.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20964.bound, LeftBound19628.bound]
def bound : CoeffClass := .finite ⟨33190381535015453579890655284, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20964.bound, LeftBound19628.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20964.actual selector witness, LeftBound19628.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20969

namespace LeftBound20973
def owner : Owner := ⟨.program ⟨214⟩, ⟨27916⟩⟩
def transferEvent : Nat := 20973
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 20971 .coefficient, .predecessor 1 20972 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20971 .coefficient)
      LeftBound20968.bound (LeftBound20968.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20970RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20968.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20968.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20972 .coefficient)
      LeftBound19414.bound (LeftBound19414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19421RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19414.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19414.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20968.bound, LeftBound19414.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20968.bound, LeftBound19414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20968.actual selector witness, LeftBound19414.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20973

namespace LeftBound20974
def owner : Owner := ⟨.program ⟨214⟩, ⟨27916⟩⟩
def transferEvent : Nat := 20974
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 20970 .summary, .result 19421 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20970 .summary)
      LeftBound20969.bound (LeftBound20969.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27699⟩⟩) (rawTerms := some (Proof.Events081.exact20970RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20969.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19421 .summary)
      LeftBound19416.bound (LeftBound19416.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27915⟩⟩) (rawTerms := some (Proof.Events075.exact19421RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound19416.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20969.bound, LeftBound19416.bound]
def bound : CoeffClass := .finite ⟨37932293507469318446662025268, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20969.bound, LeftBound19416.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20969.actual selector witness, LeftBound19416.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20974

namespace LeftBound20978
def owner : Owner := ⟨.program ⟨214⟩, ⟨28133⟩⟩
def transferEvent : Nat := 20978
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 20976 .coefficient, .predecessor 1 20977 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20976 .coefficient)
      LeftBound20973.bound (LeftBound20973.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20973.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20973.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20977 .coefficient)
      LeftBound19202.bound (LeftBound19202.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19202.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19202.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20973.bound, LeftBound19202.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20973.bound, LeftBound19202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20973.actual selector witness, LeftBound19202.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20978

namespace LeftBound20979
def owner : Owner := ⟨.program ⟨214⟩, ⟨28133⟩⟩
def transferEvent : Nat := 20979
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 20975 .summary, .result 19209 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20975 .summary)
      LeftBound20974.bound (LeftBound20974.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27916⟩⟩) (rawTerms := some (Proof.Events081.exact20975RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20974.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19209 .summary)
      LeftBound19204.bound (LeftBound19204.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28132⟩⟩) (rawTerms := some (Proof.Events075.exact19209RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound19204.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20974.bound, LeftBound19204.bound]
def bound : CoeffClass := .finite ⟨42674369987986832655214706740, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20974.bound, LeftBound19204.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20974.actual selector witness, LeftBound19204.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20979

namespace LeftBound20983
def owner : Owner := ⟨.program ⟨214⟩, ⟨28350⟩⟩
def transferEvent : Nat := 20983
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 20981 .coefficient, .predecessor 1 20982 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20981 .coefficient)
      LeftBound20978.bound (LeftBound20978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20978.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20982 .coefficient)
      LeftBound18990.bound (LeftBound18990.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact18997RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18990.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18990.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20978.bound, LeftBound18990.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20978.bound, LeftBound18990.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20978.actual selector witness, LeftBound18990.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20983

namespace LeftBound20984
def owner : Owner := ⟨.program ⟨214⟩, ⟨28350⟩⟩
def transferEvent : Nat := 20984
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 20980 .summary, .result 18997 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20980 .summary)
      LeftBound20979.bound (LeftBound20979.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28133⟩⟩) (rawTerms := some (Proof.Events081.exact20980RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18997 .summary)
      LeftBound18992.bound (LeftBound18992.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28349⟩⟩) (rawTerms := some (Proof.Events074.exact18997RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound18992.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20979.bound, LeftBound18992.bound]
def bound : CoeffClass := .finite ⟨47416693230599820876439355444, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20979.bound, LeftBound18992.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20979.actual selector witness, LeftBound18992.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20984

namespace LeftBound20988
def owner : Owner := ⟨.program ⟨214⟩, ⟨28567⟩⟩
def transferEvent : Nat := 20988
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 20986 .coefficient, .predecessor 1 20987 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20986 .coefficient)
      LeftBound20983.bound (LeftBound20983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20985RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20983.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20983.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20987 .coefficient)
      LeftBound18778.bound (LeftBound18778.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18785RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18778.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18778.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20983.bound, LeftBound18778.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20983.bound, LeftBound18778.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20983.actual selector witness, LeftBound18778.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20988

namespace LeftBound20989
def owner : Owner := ⟨.program ⟨214⟩, ⟨28567⟩⟩
def transferEvent : Nat := 20989
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 20985 .summary, .result 18785 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20985 .summary)
      LeftBound20984.bound (LeftBound20984.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28350⟩⟩) (rawTerms := some (Proof.Events081.exact20985RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20984.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18785 .summary)
      LeftBound18780.bound (LeftBound18780.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28566⟩⟩) (rawTerms := some (Proof.Events073.exact18785RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound18780.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20984.bound, LeftBound18780.bound]
def bound : CoeffClass := .finite ⟨52159098727244633768554659892, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20984.bound, LeftBound18780.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20984.actual selector witness, LeftBound18780.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20989

namespace LeftBound20993
def owner : Owner := ⟨.program ⟨214⟩, ⟨28784⟩⟩
def transferEvent : Nat := 20993
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 20991 .coefficient, .predecessor 1 20992 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20991 .coefficient)
      LeftBound20988.bound (LeftBound20988.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20990RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20988.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20988.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20992 .coefficient)
      LeftBound18566.bound (LeftBound18566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events072.exact18573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18566.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18566.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20988.bound, LeftBound18566.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20988.bound, LeftBound18566.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20988.actual selector witness, LeftBound18566.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20993

namespace LeftBound20994
def owner : Owner := ⟨.program ⟨214⟩, ⟨28784⟩⟩
def transferEvent : Nat := 20994
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 20990 .summary, .result 18573 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20990 .summary)
      LeftBound20989.bound (LeftBound20989.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28567⟩⟩) (rawTerms := some (Proof.Events081.exact20990RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20989.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18573 .summary)
      LeftBound18568.bound (LeftBound18568.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28783⟩⟩) (rawTerms := some (Proof.Events072.exact18573RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound18568.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20989.bound, LeftBound18568.bound]
def bound : CoeffClass := .finite ⟨56901750985984920673341931572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20989.bound, LeftBound18568.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20989.actual selector witness, LeftBound18568.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20994

namespace LeftBound20998
def owner : Owner := ⟨.program ⟨214⟩, ⟨29001⟩⟩
def transferEvent : Nat := 20998
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 20996 .coefficient, .predecessor 1 20997 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20996 .coefficient)
      LeftBound20993.bound (LeftBound20993.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact20995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20993.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20997 .coefficient)
      LeftBound18354.bound (LeftBound18354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events071.exact18361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18354.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20993.bound, LeftBound18354.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20993.bound, LeftBound18354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20993.actual selector witness, LeftBound18354.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20998

namespace LeftBound20999
def owner : Owner := ⟨.program ⟨214⟩, ⟨29001⟩⟩
def transferEvent : Nat := 20999
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 20995 .summary, .result 18361 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20995 .summary)
      LeftBound20994.bound (LeftBound20994.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28784⟩⟩) (rawTerms := some (Proof.Events082.exact20995RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20994.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18361 .summary)
      LeftBound18356.bound (LeftBound18356.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29000⟩⟩) (rawTerms := some (Proof.Events071.exact18361RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound18356.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20994.bound, LeftBound18356.bound]
def bound : CoeffClass := .finite ⟨61644567752788856919910514740, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20994.bound, LeftBound18356.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20994.actual selector witness, LeftBound18356.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20999

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
