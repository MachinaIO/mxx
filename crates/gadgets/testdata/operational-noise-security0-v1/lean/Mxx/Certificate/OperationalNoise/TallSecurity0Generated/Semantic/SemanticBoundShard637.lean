import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard621
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard622
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard624
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard625
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard626
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard628
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard629
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard630
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard632
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard636

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound94043
def owner : Owner := ⟨.program ⟨214⟩, ⟨26996⟩⟩
def transferEvent : Nat := 94043
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94039 .summary, .result 93361 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94039 .summary)
      LeftBound94038.bound (LeftBound94038.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26779⟩⟩) (rawTerms := some (Proof.Events367.exact94039RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94038.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93361 .summary)
      LeftBound93356.bound (LeftBound93356.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26995⟩⟩) (rawTerms := some (Proof.Events364.exact93361RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound93356.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94038.bound, LeftBound93356.bound]
def bound : CoeffClass := .finite ⟨18965303649908456346701791284, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94038.bound, LeftBound93356.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94038.actual selector witness, LeftBound93356.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94043

namespace LeftBound94047
def owner : Owner := ⟨.program ⟨214⟩, ⟨27213⟩⟩
def transferEvent : Nat := 94047
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94045 .coefficient, .predecessor 1 94046 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94045 .coefficient)
      LeftBound94042.bound (LeftBound94042.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94042.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94042.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94046 .coefficient)
      LeftBound93142.bound (LeftBound93142.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact93149RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93142.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93142.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94042.bound, LeftBound93142.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94042.bound, LeftBound93142.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94042.actual selector witness, LeftBound93142.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94047

namespace LeftBound94048
def owner : Owner := ⟨.program ⟨214⟩, ⟨27213⟩⟩
def transferEvent : Nat := 94048
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94044 .summary, .result 93149 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94044 .summary)
      LeftBound94043.bound (LeftBound94043.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26996⟩⟩) (rawTerms := some (Proof.Events367.exact94044RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94043.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93149 .summary)
      LeftBound93144.bound (LeftBound93144.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27212⟩⟩) (rawTerms := some (Proof.Events363.exact93149RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound93144.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94043.bound, LeftBound93144.bound]
def bound : CoeffClass := .finite ⟨23706886606235022529910538292, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94043.bound, LeftBound93144.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94043.actual selector witness, LeftBound93144.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94048

namespace LeftBound94052
def owner : Owner := ⟨.program ⟨214⟩, ⟨27430⟩⟩
def transferEvent : Nat := 94052
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94050 .coefficient, .predecessor 1 94051 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94050 .coefficient)
      LeftBound94047.bound (LeftBound94047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94047.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94047.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94051 .coefficient)
      LeftBound92930.bound (LeftBound92930.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact92937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92930.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92930.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94047.bound, LeftBound92930.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94047.bound, LeftBound92930.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94047.actual selector witness, LeftBound92930.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94052

namespace LeftBound94053
def owner : Owner := ⟨.program ⟨214⟩, ⟨27430⟩⟩
def transferEvent : Nat := 94053
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94049 .summary, .result 92937 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94049 .summary)
      LeftBound94048.bound (LeftBound94048.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27213⟩⟩) (rawTerms := some (Proof.Events367.exact94049RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94048.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92937 .summary)
      LeftBound92932.bound (LeftBound92932.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27429⟩⟩) (rawTerms := some (Proof.Events363.exact92937RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92932.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94048.bound, LeftBound92932.bound]
def bound : CoeffClass := .finite ⟨28448551816593413384009941044, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94048.bound, LeftBound92932.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94048.actual selector witness, LeftBound92932.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94053

namespace LeftBound94057
def owner : Owner := ⟨.program ⟨214⟩, ⟨27647⟩⟩
def transferEvent : Nat := 94057
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94055 .coefficient, .predecessor 1 94056 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94055 .coefficient)
      LeftBound94052.bound (LeftBound94052.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94054RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94052.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94052.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94056 .coefficient)
      LeftBound92718.bound (LeftBound92718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92718.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92718.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94052.bound, LeftBound92718.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94052.bound, LeftBound92718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94052.actual selector witness, LeftBound92718.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94057

namespace LeftBound94058
def owner : Owner := ⟨.program ⟨214⟩, ⟨27647⟩⟩
def transferEvent : Nat := 94058
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94054 .summary, .result 92725 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94054 .summary)
      LeftBound94053.bound (LeftBound94053.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27430⟩⟩) (rawTerms := some (Proof.Events367.exact94054RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94053.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92725 .summary)
      LeftBound92720.bound (LeftBound92720.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27646⟩⟩) (rawTerms := some (Proof.Events362.exact92725RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92720.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94053.bound, LeftBound92720.bound]
def bound : CoeffClass := .finite ⟨33190381535015453579890655284, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94053.bound, LeftBound92720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94053.actual selector witness, LeftBound92720.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94058

namespace LeftBound94062
def owner : Owner := ⟨.program ⟨214⟩, ⟨27864⟩⟩
def transferEvent : Nat := 94062
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94060 .coefficient, .predecessor 1 94061 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94060 .coefficient)
      LeftBound94057.bound (LeftBound94057.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94057.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94057.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94061 .coefficient)
      LeftBound92506.bound (LeftBound92506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92506.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94057.bound, LeftBound92506.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94057.bound, LeftBound92506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94057.actual selector witness, LeftBound92506.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94062

namespace LeftBound94063
def owner : Owner := ⟨.program ⟨214⟩, ⟨27864⟩⟩
def transferEvent : Nat := 94063
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94059 .summary, .result 92513 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94059 .summary)
      LeftBound94058.bound (LeftBound94058.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27647⟩⟩) (rawTerms := some (Proof.Events367.exact94059RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94058.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92513 .summary)
      LeftBound92508.bound (LeftBound92508.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27863⟩⟩) (rawTerms := some (Proof.Events361.exact92513RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92508.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94058.bound, LeftBound92508.bound]
def bound : CoeffClass := .finite ⟨37932293507469318446662025268, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94058.bound, LeftBound92508.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94058.actual selector witness, LeftBound92508.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94063

namespace LeftBound94067
def owner : Owner := ⟨.program ⟨214⟩, ⟨28081⟩⟩
def transferEvent : Nat := 94067
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94065 .coefficient, .predecessor 1 94066 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94065 .coefficient)
      LeftBound94062.bound (LeftBound94062.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94062.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94062.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94066 .coefficient)
      LeftBound92294.bound (LeftBound92294.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events360.exact92301RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92294.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92294.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94062.bound, LeftBound92294.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94062.bound, LeftBound92294.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94062.actual selector witness, LeftBound92294.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94067

namespace LeftBound94068
def owner : Owner := ⟨.program ⟨214⟩, ⟨28081⟩⟩
def transferEvent : Nat := 94068
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94064 .summary, .result 92301 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94064 .summary)
      LeftBound94063.bound (LeftBound94063.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27864⟩⟩) (rawTerms := some (Proof.Events367.exact94064RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94063.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92301 .summary)
      LeftBound92296.bound (LeftBound92296.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28080⟩⟩) (rawTerms := some (Proof.Events360.exact92301RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92296.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94063.bound, LeftBound92296.bound]
def bound : CoeffClass := .finite ⟨42674369987986832655214706740, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94063.bound, LeftBound92296.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94063.actual selector witness, LeftBound92296.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94068

namespace LeftBound94072
def owner : Owner := ⟨.program ⟨214⟩, ⟨28298⟩⟩
def transferEvent : Nat := 94072
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94070 .coefficient, .predecessor 1 94071 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94070 .coefficient)
      LeftBound94067.bound (LeftBound94067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94069RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94071 .coefficient)
      LeftBound92082.bound (LeftBound92082.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events359.exact92089RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92082.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92082.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94067.bound, LeftBound92082.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94067.bound, LeftBound92082.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94067.actual selector witness, LeftBound92082.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94072

namespace LeftBound94073
def owner : Owner := ⟨.program ⟨214⟩, ⟨28298⟩⟩
def transferEvent : Nat := 94073
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94069 .summary, .result 92089 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94069 .summary)
      LeftBound94068.bound (LeftBound94068.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28081⟩⟩) (rawTerms := some (Proof.Events367.exact94069RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94068.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92089 .summary)
      LeftBound92084.bound (LeftBound92084.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28297⟩⟩) (rawTerms := some (Proof.Events359.exact92089RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92084.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94068.bound, LeftBound92084.bound]
def bound : CoeffClass := .finite ⟨47416693230599820876439355444, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94068.bound, LeftBound92084.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94068.actual selector witness, LeftBound92084.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94073

namespace LeftBound94077
def owner : Owner := ⟨.program ⟨214⟩, ⟨28515⟩⟩
def transferEvent : Nat := 94077
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94075 .coefficient, .predecessor 1 94076 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94075 .coefficient)
      LeftBound94072.bound (LeftBound94072.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94074RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94072.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94072.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94076 .coefficient)
      LeftBound91870.bound (LeftBound91870.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91877RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91870.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91870.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94072.bound, LeftBound91870.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94072.bound, LeftBound91870.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94072.actual selector witness, LeftBound91870.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94077

namespace LeftBound94078
def owner : Owner := ⟨.program ⟨214⟩, ⟨28515⟩⟩
def transferEvent : Nat := 94078
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94074 .summary, .result 91877 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94074 .summary)
      LeftBound94073.bound (LeftBound94073.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28298⟩⟩) (rawTerms := some (Proof.Events367.exact94074RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94073.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91877 .summary)
      LeftBound91872.bound (LeftBound91872.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28514⟩⟩) (rawTerms := some (Proof.Events358.exact91877RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91872.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94073.bound, LeftBound91872.bound]
def bound : CoeffClass := .finite ⟨52159098727244633768554659892, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94073.bound, LeftBound91872.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94073.actual selector witness, LeftBound91872.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94078

namespace LeftBound94082
def owner : Owner := ⟨.program ⟨214⟩, ⟨28732⟩⟩
def transferEvent : Nat := 94082
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94080 .coefficient, .predecessor 1 94081 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94080 .coefficient)
      LeftBound94077.bound (LeftBound94077.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94079RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94077.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94077.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94081 .coefficient)
      LeftBound91658.bound (LeftBound91658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91658.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94077.bound, LeftBound91658.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94077.bound, LeftBound91658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94077.actual selector witness, LeftBound91658.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94082

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
