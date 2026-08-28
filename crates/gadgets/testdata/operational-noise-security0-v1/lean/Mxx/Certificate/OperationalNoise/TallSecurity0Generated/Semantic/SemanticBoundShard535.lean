import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard526
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard527
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard528
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard529
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard530
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard531
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard532
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard533
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard534

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound79431
def owner : Owner := ⟨.program ⟨214⟩, ⟨7807⟩⟩
def transferEvent : Nat := 79431
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79429 .coefficient, .predecessor 1 79430 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79429 .coefficient)
      LeftBound79425.bound (LeftBound79425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79428RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79425.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79425.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79430 .coefficient)
      LeftBound79425.bound (LeftBound79425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79428RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79425.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79425.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79425.bound, LeftBound79425.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79425.bound, LeftBound79425.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79425.actual selector witness, LeftBound79425.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79431

namespace LeftBound79434
def owner : Owner := ⟨.program ⟨214⟩, ⟨7807⟩⟩
def transferEvent : Nat := 79434
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79428 .summary, .result 79428 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79428 .summary)
      LeftBound79426.bound (LeftBound79426.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7750⟩⟩) (rawTerms := some (Proof.Events310.exact79428RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79426.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79428 .summary)
      LeftBound79426.bound (LeftBound79426.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7750⟩⟩) (rawTerms := some (Proof.Events310.exact79428RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79426.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79426.bound, LeftBound79426.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79426.bound, LeftBound79426.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79426.actual selector witness, LeftBound79426.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79434

namespace LeftBound79438
def owner : Owner := ⟨.program ⟨214⟩, ⟨26344⟩⟩
def transferEvent : Nat := 79438
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79436 .coefficient, .predecessor 1 79437 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79436 .coefficient)
      LeftBound79431.bound (LeftBound79431.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79435RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79431.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79431.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79437 .coefficient)
      LeftBound79401.bound (LeftBound79401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79408RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79401.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79401.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79431.bound, LeftBound79401.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79431.bound, LeftBound79401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79431.actual selector witness, LeftBound79401.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79438

namespace LeftBound79439
def owner : Owner := ⟨.program ⟨214⟩, ⟨26344⟩⟩
def transferEvent : Nat := 79439
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79435 .summary, .result 79408 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79435 .summary)
      LeftBound79434.bound (LeftBound79434.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7807⟩⟩) (rawTerms := some (Proof.Events310.exact79435RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79434.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79408 .summary)
      LeftBound79403.bound (LeftBound79403.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26343⟩⟩) (rawTerms := some (Proof.Events310.exact79408RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79403.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79434.bound, LeftBound79403.bound]
def bound : CoeffClass := .finite ⟨4741253940199267499646124084, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79434.bound, LeftBound79403.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79434.actual selector witness, LeftBound79403.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79439

namespace LeftBound79443
def owner : Owner := ⟨.program ⟨214⟩, ⟨26549⟩⟩
def transferEvent : Nat := 79443
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79441 .coefficient, .predecessor 1 79442 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79441 .coefficient)
      LeftBound79438.bound (LeftBound79438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79440RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79438.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79438.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79442 .coefficient)
      LeftBound79189.bound (LeftBound79189.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79196RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79189.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79189.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79438.bound, LeftBound79189.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79438.bound, LeftBound79189.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79438.actual selector witness, LeftBound79189.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79443

namespace LeftBound79444
def owner : Owner := ⟨.program ⟨214⟩, ⟨26549⟩⟩
def transferEvent : Nat := 79444
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79440 .summary, .result 79196 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79440 .summary)
      LeftBound79439.bound (LeftBound79439.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26344⟩⟩) (rawTerms := some (Proof.Events310.exact79440RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79439.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79196 .summary)
      LeftBound79191.bound (LeftBound79191.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26548⟩⟩) (rawTerms := some (Proof.Events309.exact79196RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79191.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79439.bound, LeftBound79191.bound]
def bound : CoeffClass := .finite ⟨9482549007414447334737575988, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79439.bound, LeftBound79191.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79439.actual selector witness, LeftBound79191.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79444

namespace LeftBound79448
def owner : Owner := ⟨.program ⟨214⟩, ⟨26766⟩⟩
def transferEvent : Nat := 79448
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79446 .coefficient, .predecessor 1 79447 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79446 .coefficient)
      LeftBound79443.bound (LeftBound79443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79445RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79443.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79443.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79447 .coefficient)
      LeftBound78977.bound (LeftBound78977.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events308.exact78984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78977.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78977.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79443.bound, LeftBound78977.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79443.bound, LeftBound78977.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79443.actual selector witness, LeftBound78977.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79448

namespace LeftBound79449
def owner : Owner := ⟨.program ⟨214⟩, ⟨26766⟩⟩
def transferEvent : Nat := 79449
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79445 .summary, .result 78984 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79445 .summary)
      LeftBound79444.bound (LeftBound79444.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26549⟩⟩) (rawTerms := some (Proof.Events310.exact79445RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79444.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78984 .summary)
      LeftBound78979.bound (LeftBound78979.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26765⟩⟩) (rawTerms := some (Proof.Events308.exact78984RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound78979.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79444.bound, LeftBound78979.bound]
def bound : CoeffClass := .finite ⟨14223885201645539505274355764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79444.bound, LeftBound78979.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79444.actual selector witness, LeftBound78979.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79449

namespace LeftBound79453
def owner : Owner := ⟨.program ⟨214⟩, ⟨26983⟩⟩
def transferEvent : Nat := 79453
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79451 .coefficient, .predecessor 1 79452 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79451 .coefficient)
      LeftBound79448.bound (LeftBound79448.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79450RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79448.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79448.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79452 .coefficient)
      LeftBound78765.bound (LeftBound78765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events307.exact78772RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78765.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78765.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79448.bound, LeftBound78765.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79448.bound, LeftBound78765.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79448.actual selector witness, LeftBound78765.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79453

namespace LeftBound79454
def owner : Owner := ⟨.program ⟨214⟩, ⟨26983⟩⟩
def transferEvent : Nat := 79454
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79450 .summary, .result 78772 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79450 .summary)
      LeftBound79449.bound (LeftBound79449.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26766⟩⟩) (rawTerms := some (Proof.Events310.exact79450RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79449.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78772 .summary)
      LeftBound78767.bound (LeftBound78767.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26982⟩⟩) (rawTerms := some (Proof.Events307.exact78772RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound78767.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79449.bound, LeftBound78767.bound]
def bound : CoeffClass := .finite ⟨18965303649908456346701791284, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79449.bound, LeftBound78767.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79449.actual selector witness, LeftBound78767.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79454

namespace LeftBound79458
def owner : Owner := ⟨.program ⟨214⟩, ⟨27200⟩⟩
def transferEvent : Nat := 79458
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79456 .coefficient, .predecessor 1 79457 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79456 .coefficient)
      LeftBound79453.bound (LeftBound79453.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79455RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79453.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79453.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79457 .coefficient)
      LeftBound78553.bound (LeftBound78553.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events306.exact78560RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78553.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78553.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79453.bound, LeftBound78553.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79453.bound, LeftBound78553.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79453.actual selector witness, LeftBound78553.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79458

namespace LeftBound79459
def owner : Owner := ⟨.program ⟨214⟩, ⟨27200⟩⟩
def transferEvent : Nat := 79459
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79455 .summary, .result 78560 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79455 .summary)
      LeftBound79454.bound (LeftBound79454.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26983⟩⟩) (rawTerms := some (Proof.Events310.exact79455RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79454.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78560 .summary)
      LeftBound78555.bound (LeftBound78555.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27199⟩⟩) (rawTerms := some (Proof.Events306.exact78560RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound78555.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79454.bound, LeftBound78555.bound]
def bound : CoeffClass := .finite ⟨23706886606235022529910538292, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79454.bound, LeftBound78555.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79454.actual selector witness, LeftBound78555.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79459

namespace LeftBound79463
def owner : Owner := ⟨.program ⟨214⟩, ⟨27417⟩⟩
def transferEvent : Nat := 79463
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79461 .coefficient, .predecessor 1 79462 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79461 .coefficient)
      LeftBound79458.bound (LeftBound79458.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79458.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79458.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79462 .coefficient)
      LeftBound78341.bound (LeftBound78341.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events306.exact78348RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78341.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78341.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79458.bound, LeftBound78341.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79458.bound, LeftBound78341.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79458.actual selector witness, LeftBound78341.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79463

namespace LeftBound79464
def owner : Owner := ⟨.program ⟨214⟩, ⟨27417⟩⟩
def transferEvent : Nat := 79464
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79460 .summary, .result 78348 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79460 .summary)
      LeftBound79459.bound (LeftBound79459.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27200⟩⟩) (rawTerms := some (Proof.Events310.exact79460RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78348 .summary)
      LeftBound78343.bound (LeftBound78343.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27416⟩⟩) (rawTerms := some (Proof.Events306.exact78348RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound78343.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79459.bound, LeftBound78343.bound]
def bound : CoeffClass := .finite ⟨28448551816593413384009941044, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79459.bound, LeftBound78343.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79459.actual selector witness, LeftBound78343.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79464

namespace LeftBound79468
def owner : Owner := ⟨.program ⟨214⟩, ⟨27634⟩⟩
def transferEvent : Nat := 79468
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79466 .coefficient, .predecessor 1 79467 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79466 .coefficient)
      LeftBound79463.bound (LeftBound79463.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79465RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79463.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79463.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79467 .coefficient)
      LeftBound78129.bound (LeftBound78129.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78129.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78129.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79463.bound, LeftBound78129.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79463.bound, LeftBound78129.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79463.actual selector witness, LeftBound78129.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79468

namespace LeftBound79469
def owner : Owner := ⟨.program ⟨214⟩, ⟨27634⟩⟩
def transferEvent : Nat := 79469
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79465 .summary, .result 78136 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79465 .summary)
      LeftBound79464.bound (LeftBound79464.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27417⟩⟩) (rawTerms := some (Proof.Events310.exact79465RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79464.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78136 .summary)
      LeftBound78131.bound (LeftBound78131.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27633⟩⟩) (rawTerms := some (Proof.Events305.exact78136RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound78131.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79464.bound, LeftBound78131.bound]
def bound : CoeffClass := .finite ⟨33190381535015453579890655284, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79464.bound, LeftBound78131.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79464.actual selector witness, LeftBound78131.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79469

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
