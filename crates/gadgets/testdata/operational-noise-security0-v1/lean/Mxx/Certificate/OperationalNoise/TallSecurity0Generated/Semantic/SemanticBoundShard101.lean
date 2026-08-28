import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard032
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard036
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard040
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard044
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard048
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard052
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard100

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound15513
def owner : Owner := ⟨.program ⟨214⟩, ⟨28790⟩⟩
def transferEvent : Nat := 15513
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 15509 .summary, .result 9948 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15509 .summary)
      LeftBound15508.bound (LeftBound15508.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28573⟩⟩) (rawTerms := some (Proof.Events060.exact15509RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15508.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9948 .summary)
      LeftBound9947.bound (LeftBound9947.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28789⟩⟩) (rawTerms := some (Proof.Events038.exact9948RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound9947.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15508.bound, LeftBound9947.bound]
def bound : CoeffClass := .finite ⟨15504496706822237470720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15508.bound, LeftBound9947.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15508.actual selector witness, LeftBound9947.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15513

namespace LeftBound15517
def owner : Owner := ⟨.program ⟨214⟩, ⟨29007⟩⟩
def transferEvent : Nat := 15517
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15515 .coefficient, .predecessor 1 15516 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15515 .coefficient)
      LeftBound15512.bound (LeftBound15512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15514RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15512.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15516 .coefficient)
      LeftBound9443.bound (LeftBound9443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9447RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9443.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9443.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15512.bound, LeftBound9443.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15512.bound, LeftBound9443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15512.actual selector witness, LeftBound9443.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15517

namespace LeftBound15518
def owner : Owner := ⟨.program ⟨214⟩, ⟨29007⟩⟩
def transferEvent : Nat := 15518
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 15514 .summary, .result 9447 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15514 .summary)
      LeftBound15513.bound (LeftBound15513.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28790⟩⟩) (rawTerms := some (Proof.Events060.exact15514RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15513.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9447 .summary)
      LeftBound9446.bound (LeftBound9446.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29006⟩⟩) (rawTerms := some (Proof.Events036.exact9447RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound9446.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15513.bound, LeftBound9446.bound]
def bound : CoeffClass := .finite ⟨16796811717657050247168, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15513.bound, LeftBound9446.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15513.actual selector witness, LeftBound9446.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15518

namespace LeftBound15522
def owner : Owner := ⟨.program ⟨214⟩, ⟨29224⟩⟩
def transferEvent : Nat := 15522
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15520 .coefficient, .predecessor 1 15521 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15520 .coefficient)
      LeftBound15517.bound (LeftBound15517.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15519RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15517.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15517.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15521 .coefficient)
      LeftBound8942.bound (LeftBound8942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8946RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8942.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8942.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15517.bound, LeftBound8942.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15517.bound, LeftBound8942.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15517.actual selector witness, LeftBound8942.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15522

namespace LeftBound15523
def owner : Owner := ⟨.program ⟨214⟩, ⟨29224⟩⟩
def transferEvent : Nat := 15523
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 15519 .summary, .result 8946 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15519 .summary)
      LeftBound15518.bound (LeftBound15518.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29007⟩⟩) (rawTerms := some (Proof.Events060.exact15519RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15518.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8946 .summary)
      LeftBound8945.bound (LeftBound8945.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29223⟩⟩) (rawTerms := some (Proof.Events034.exact8946RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound8945.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15518.bound, LeftBound8945.bound]
def bound : CoeffClass := .finite ⟨18089149140936883609600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15518.bound, LeftBound8945.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15518.actual selector witness, LeftBound8945.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15523

namespace LeftBound15527
def owner : Owner := ⟨.program ⟨214⟩, ⟨29441⟩⟩
def transferEvent : Nat := 15527
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15525 .coefficient, .predecessor 1 15526 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15525 .coefficient)
      LeftBound15522.bound (LeftBound15522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15524RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15522.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15522.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15526 .coefficient)
      LeftBound8441.bound (LeftBound8441.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8445RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8441.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8441.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15522.bound, LeftBound8441.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15522.bound, LeftBound8441.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15522.actual selector witness, LeftBound8441.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15527

namespace LeftBound15528
def owner : Owner := ⟨.program ⟨214⟩, ⟨29441⟩⟩
def transferEvent : Nat := 15528
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 15524 .summary, .result 8445 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15524 .summary)
      LeftBound15523.bound (LeftBound15523.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29224⟩⟩) (rawTerms := some (Proof.Events060.exact15524RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15523.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8445 .summary)
      LeftBound8444.bound (LeftBound8444.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29440⟩⟩) (rawTerms := some (Proof.Events032.exact8445RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound8444.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15523.bound, LeftBound8444.bound]
def bound : CoeffClass := .finite ⟨19381531389106758144000, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15523.bound, LeftBound8444.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15523.actual selector witness, LeftBound8444.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15528

namespace LeftBound15532
def owner : Owner := ⟨.program ⟨214⟩, ⟨29658⟩⟩
def transferEvent : Nat := 15532
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15530 .coefficient, .predecessor 1 15531 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15530 .coefficient)
      LeftBound15527.bound (LeftBound15527.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15529RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15527.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15527.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15531 .coefficient)
      LeftBound7940.bound (LeftBound7940.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact7944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7940.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7940.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15527.bound, LeftBound7940.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15527.bound, LeftBound7940.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15527.actual selector witness, LeftBound7940.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15532

namespace LeftBound15533
def owner : Owner := ⟨.program ⟨214⟩, ⟨29658⟩⟩
def transferEvent : Nat := 15533
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 15529 .summary, .result 7944 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15529 .summary)
      LeftBound15528.bound (LeftBound15528.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29441⟩⟩) (rawTerms := some (Proof.Events060.exact15529RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15528.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7944 .summary)
      LeftBound7943.bound (LeftBound7943.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29657⟩⟩) (rawTerms := some (Proof.Events031.exact7944RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound7943.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15528.bound, LeftBound7943.bound]
def bound : CoeffClass := .finite ⟨20673980874611694436352, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15528.bound, LeftBound7943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15528.actual selector witness, LeftBound7943.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15533

namespace LeftBound15537
def owner : Owner := ⟨.program ⟨214⟩, ⟨29875⟩⟩
def transferEvent : Nat := 15537
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15535 .coefficient, .predecessor 1 15536 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15535 .coefficient)
      LeftBound15532.bound (LeftBound15532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15534RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15532.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15532.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15536 .coefficient)
      LeftBound7439.bound (LeftBound7439.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7443RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7439.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7439.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15532.bound, LeftBound7439.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15532.bound, LeftBound7439.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15532.actual selector witness, LeftBound7439.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15537

namespace LeftBound15538
def owner : Owner := ⟨.program ⟨214⟩, ⟨29875⟩⟩
def transferEvent : Nat := 15538
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 15534 .summary, .result 7443 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15534 .summary)
      LeftBound15533.bound (LeftBound15533.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29658⟩⟩) (rawTerms := some (Proof.Events060.exact15534RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15533.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7443 .summary)
      LeftBound7442.bound (LeftBound7442.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29874⟩⟩) (rawTerms := some (Proof.Events029.exact7443RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound7442.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15533.bound, LeftBound7442.bound]
def bound : CoeffClass := .finite ⟨21966497597451692486656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15533.bound, LeftBound7442.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15533.actual selector witness, LeftBound7442.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15538

namespace LeftBound15542
def owner : Owner := ⟨.program ⟨214⟩, ⟨30209⟩⟩
def transferEvent : Nat := 15542
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 15540 .coefficient, .predecessor 1 15541 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15540 .coefficient)
      LeftBound15537.bound (LeftBound15537.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15539RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15537.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15537.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15541 .coefficient)
      LeftBound6938.bound (LeftBound6938.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6942RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6938.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6938.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15537.bound, LeftBound6938.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15537.bound, LeftBound6938.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15537.actual selector witness, LeftBound6938.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15542

namespace LeftBound15543
def owner : Owner := ⟨.program ⟨214⟩, ⟨30209⟩⟩
def transferEvent : Nat := 15543
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 15539 .summary, .result 6942 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15539 .summary)
      LeftBound15538.bound (LeftBound15538.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29875⟩⟩) (rawTerms := some (Proof.Events060.exact15539RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15538.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6942 .summary)
      LeftBound6941.bound (LeftBound6941.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30208⟩⟩) (rawTerms := some (Proof.Events027.exact6942RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6941.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound15538.bound, LeftBound6941.bound]
def bound : CoeffClass := .finite ⟨23259036732736711122944, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15538.bound, LeftBound6941.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound15538.actual selector witness, LeftBound6941.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound15543

namespace LeftBound15547
def owner : Owner := ⟨.program ⟨214⟩, ⟨30210⟩⟩
def transferEvent : Nat := 15547
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 15545 .coefficient) (.predecessor 1 15546 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15545 .coefficient)
      LeftBound15542.bound (LeftBound15542.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15544RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15542.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15542.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15546 .coefficient)
      LeftAuthority6418.bound (LeftAuthority6418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6419RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6418.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6418.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound15542.bound LeftAuthority6418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15542.bound, LeftAuthority6418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound15542.actual selector witness) * (LeftAuthority6418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15547

namespace LeftBound15548
def owner : Owner := ⟨.program ⟨214⟩, ⟨30210⟩⟩
def transferEvent : Nat := 15548
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩ [⟨.result 6419 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6419 .coefficient)
      LeftAuthority6418.bound (LeftAuthority6418.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18693⟩⟩) (rawTerms := some (Proof.Events025.exact6419RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6418.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6418.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6418.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6418.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound15548

namespace LeftBound15549
def owner : Owner := ⟨.program ⟨214⟩, ⟨30210⟩⟩
def transferEvent : Nat := 15549
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 15544 .summary) (.transfer 15548) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15544 .summary)
      LeftBound15543.bound (LeftBound15543.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30209⟩⟩) (rawTerms := some (Proof.Events060.exact15544RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15543.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 15548)
      LeftBound15548.bound (LeftBound15548.actual selector witness) := by
  exact .transfer (LeftBound15548.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound15543.bound LeftBound15548.bound
def bound : CoeffClass := .finite ⟨85361036953731453608582447104, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15543.bound, LeftBound15548.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound15543.actual selector witness) * (LeftBound15548.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15549

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
