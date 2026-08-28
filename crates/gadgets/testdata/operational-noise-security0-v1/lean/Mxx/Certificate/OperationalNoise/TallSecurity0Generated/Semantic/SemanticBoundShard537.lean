import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard022
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard510
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard511
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard512
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard513
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard514
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard536

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound79513
def owner : Owner := ⟨.program ⟨214⟩, ⟨29587⟩⟩
def transferEvent : Nat := 79513
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79511 .coefficient, .predecessor 1 79512 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79511 .coefficient)
      LeftBound79508.bound (LeftBound79508.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79508.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79512 .coefficient)
      LeftBound76221.bound (LeftBound76221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76228RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76221.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76221.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79508.bound, LeftBound76221.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79508.bound, LeftBound76221.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79508.actual selector witness, LeftBound76221.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79513

namespace LeftBound79514
def owner : Owner := ⟨.program ⟨214⟩, ⟨29587⟩⟩
def transferEvent : Nat := 79514
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79510 .summary, .result 76228 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79510 .summary)
      LeftBound79509.bound (LeftBound79509.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29370⟩⟩) (rawTerms := some (Proof.Events310.exact79510RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76228 .summary)
      LeftBound76223.bound (LeftBound76223.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29586⟩⟩) (rawTerms := some (Proof.Events297.exact76228RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76223.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79509.bound, LeftBound76223.bound]
def bound : CoeffClass := .finite ⟨75873840593518912368522821684, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79509.bound, LeftBound76223.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79509.actual selector witness, LeftBound76223.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79514

namespace LeftBound79518
def owner : Owner := ⟨.program ⟨214⟩, ⟨29804⟩⟩
def transferEvent : Nat := 79518
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79516 .coefficient, .predecessor 1 79517 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79516 .coefficient)
      LeftBound79513.bound (LeftBound79513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79513.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79513.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79517 .coefficient)
      LeftBound76009.bound (LeftBound76009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact76016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76009.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79513.bound, LeftBound76009.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79513.bound, LeftBound76009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79513.actual selector witness, LeftBound76009.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79518

namespace LeftBound79519
def owner : Owner := ⟨.program ⟨214⟩, ⟨29804⟩⟩
def transferEvent : Nat := 79519
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79515 .summary, .result 76016 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79515 .summary)
      LeftBound79514.bound (LeftBound79514.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29587⟩⟩) (rawTerms := some (Proof.Events310.exact79515RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79514.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76016 .summary)
      LeftBound76011.bound (LeftBound76011.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29803⟩⟩) (rawTerms := some (Proof.Events296.exact76016RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76011.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79514.bound, LeftBound76011.bound]
def bound : CoeffClass := .finite ⟨80617397646609270653107306548, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79514.bound, LeftBound76011.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79514.actual selector witness, LeftBound76011.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79519

namespace LeftBound79523
def owner : Owner := ⟨.program ⟨214⟩, ⟨30093⟩⟩
def transferEvent : Nat := 79523
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79521 .coefficient, .predecessor 1 79522 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79521 .coefficient)
      LeftBound79518.bound (LeftBound79518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79518.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79522 .coefficient)
      LeftBound75797.bound (LeftBound75797.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75804RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75797.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75797.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79518.bound, LeftBound75797.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79518.bound, LeftBound75797.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79518.actual selector witness, LeftBound75797.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79523

namespace LeftBound79524
def owner : Owner := ⟨.program ⟨214⟩, ⟨30093⟩⟩
def transferEvent : Nat := 79524
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79520 .summary, .result 75804 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79520 .summary)
      LeftBound79519.bound (LeftBound79519.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29804⟩⟩) (rawTerms := some (Proof.Events310.exact79520RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79519.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 75804 .summary)
      LeftBound75799.bound (LeftBound75799.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30092⟩⟩) (rawTerms := some (Proof.Events296.exact75804RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75799.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79519.bound, LeftBound75799.bound]
def bound : CoeffClass := .finite ⟨85361036953731453608582447156, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79519.bound, LeftBound75799.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79519.actual selector witness, LeftBound75799.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79524

namespace LeftBound79528
def owner : Owner := ⟨.program ⟨214⟩, ⟨30104⟩⟩
def transferEvent : Nat := 79528
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79526 .coefficient, .predecessor 1 79527 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79526 .coefficient)
      LeftBound79523.bound (LeftBound79523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79523.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79523.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79527 .coefficient)
      LeftBound75585.bound (LeftBound75585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75585.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75585.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79523.bound, LeftBound75585.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79523.bound, LeftBound75585.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79523.actual selector witness, LeftBound75585.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79528

namespace LeftBound79529
def owner : Owner := ⟨.program ⟨214⟩, ⟨30104⟩⟩
def transferEvent : Nat := 79529
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79525 .summary, .result 75592 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79525 .summary)
      LeftBound79524.bound (LeftBound79524.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30093⟩⟩) (rawTerms := some (Proof.Events310.exact79525RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79524.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 75592 .summary)
      LeftBound75587.bound (LeftBound75587.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30102⟩⟩) (rawTerms := some (Proof.Events295.exact75592RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75587.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79524.bound, LeftBound75587.bound]
def bound : CoeffClass := .finite ⟨313276456757822654825721789388161076, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79524.bound, LeftBound75587.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79524.actual selector witness, LeftBound75587.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79529

namespace LeftBound79535
def owner : Owner := ⟨.program ⟨214⟩, ⟨7093⟩⟩
def transferEvent : Nat := 79535
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 79533 .coefficient) (.predecessor 1 79534 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79533 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79534 .coefficient)
      LeftAuthority6123.bound (LeftAuthority6123.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6124RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6123.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6123.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftAuthority6123.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftAuthority6123.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftAuthority6123.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79535

namespace LeftBound79540
def owner : Owner := ⟨.program ⟨214⟩, ⟨7727⟩⟩
def transferEvent : Nat := 79540
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79538 .coefficient, .predecessor 1 79539 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79538 .coefficient)
      LeftBound79535.bound (LeftBound79535.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79537RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79535.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79535.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79539 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79535.bound, LeftBound65293.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79535.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79535.actual selector witness, LeftBound65293.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79540

namespace LeftBound79544
def owner : Owner := ⟨.program ⟨214⟩, ⟨7728⟩⟩
def transferEvent : Nat := 79544
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79542 .coefficient, .predecessor 1 79543 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79542 .coefficient)
      LeftBound79540.bound (LeftBound79540.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79541RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79540.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79540.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79543 .coefficient)
      LeftAuthority79531.bound (LeftAuthority79531.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79532RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79531.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79531.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79540.bound, LeftAuthority79531.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79540.bound, LeftAuthority79531.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79540.actual selector witness, LeftAuthority79531.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79544

namespace LeftBound79545
def owner : Owner := ⟨.program ⟨214⟩, ⟨7728⟩⟩
def transferEvent : Nat := 79545
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨55⟩⟩]⟩ [⟨.result 79532 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79532 .coefficient)
      LeftAuthority79531.bound (LeftAuthority79531.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨55⟩⟩) (rawTerms := some (Proof.Events310.exact79532RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79531.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79531.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority79531.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79531.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority79531.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound79545

namespace LeftBound79550
def owner : Owner := ⟨.program ⟨214⟩, ⟨7903⟩⟩
def transferEvent : Nat := 79550
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 79548 .coefficient) (.predecessor 1 79549 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79548 .coefficient)
      LeftBound79544.bound (LeftBound79544.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79547RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79544.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79544.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79549 .coefficient)
      LeftBound5960.bound (LeftBound5960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5960.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound79544.bound LeftBound5960.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79544.bound, LeftBound5960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound79544.actual selector witness) * (LeftBound5960.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79550

namespace LeftBound79551
def owner : Owner := ⟨.program ⟨214⟩, ⟨7903⟩⟩
def transferEvent : Nat := 79551
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩ [⟨.result 5957 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5957 .coefficient)
      LeftAuthority5956.bound (LeftAuthority5956.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7885⟩⟩) (rawTerms := some (Proof.Events023.exact5957RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5956.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5956.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5956.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5956.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5956.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound79551

namespace LeftBound79552
def owner : Owner := ⟨.program ⟨214⟩, ⟨7903⟩⟩
def transferEvent : Nat := 79552
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 79547 .summary) (.transfer 79551) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79547 .summary)
      LeftBound79545.bound (LeftBound79545.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7728⟩⟩) (rawTerms := some (Proof.Events310.exact79547RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79545.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79551)
      LeftBound79551.bound (LeftBound79551.actual selector witness) := by
  exact .transfer (LeftBound79551.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound79545.bound LeftBound79551.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79545.bound, LeftBound79551.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound79545.actual selector witness) * (LeftBound79551.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79552

namespace LeftBound79578
def owner : Owner := ⟨.program ⟨214⟩, ⟨30105⟩⟩
def transferEvent : Nat := 79578
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79576 .coefficient, .predecessor 1 79577 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79576 .coefficient)
      LeftBound79550.bound (LeftBound79550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79575RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79550.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79550.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79577 .coefficient)
      LeftBound79528.bound (LeftBound79528.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events310.exact79530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79528.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79528.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79550.bound, LeftBound79528.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79550.bound, LeftBound79528.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79550.actual selector witness, LeftBound79528.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79578

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
