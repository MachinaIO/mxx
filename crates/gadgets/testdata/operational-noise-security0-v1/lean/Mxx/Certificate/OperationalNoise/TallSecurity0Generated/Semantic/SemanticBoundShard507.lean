import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard443
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard447
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard451
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard454
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard458
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard506

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound74025
def owner : Owner := ⟨.program ⟨214⟩, ⟨29159⟩⟩
def transferEvent : Nat := 74025
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 74023 .coefficient, .predecessor 1 74024 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 74023 .coefficient)
      LeftBound74020.bound (LeftBound74020.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events289.exact74022RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74020.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74020.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 74024 .coefficient)
      LeftBound67692.bound (LeftBound67692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67696RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67692.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74020.bound, LeftBound67692.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74020.bound, LeftBound67692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound74020.actual selector witness, LeftBound67692.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74025

namespace LeftBound74026
def owner : Owner := ⟨.program ⟨214⟩, ⟨29159⟩⟩
def transferEvent : Nat := 74026
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 74022 .summary, .result 67696 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 74022 .summary)
      LeftBound74021.bound (LeftBound74021.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28942⟩⟩) (rawTerms := some (Proof.Events289.exact74022RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound74021.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67696 .summary)
      LeftBound67695.bound (LeftBound67695.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29158⟩⟩) (rawTerms := some (Proof.Events264.exact67696RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67695.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74021.bound, LeftBound67695.bound]
def bound : CoeffClass := .finite ⟨18089149140936883609600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74021.bound, LeftBound67695.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound74021.actual selector witness, LeftBound67695.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74026

namespace LeftBound74030
def owner : Owner := ⟨.program ⟨214⟩, ⟨29376⟩⟩
def transferEvent : Nat := 74030
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 74028 .coefficient, .predecessor 1 74029 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 74028 .coefficient)
      LeftBound74025.bound (LeftBound74025.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events289.exact74027RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74025.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74025.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 74029 .coefficient)
      LeftBound67210.bound (LeftBound67210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67214RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67210.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67210.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74025.bound, LeftBound67210.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74025.bound, LeftBound67210.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound74025.actual selector witness, LeftBound67210.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74030

namespace LeftBound74031
def owner : Owner := ⟨.program ⟨214⟩, ⟨29376⟩⟩
def transferEvent : Nat := 74031
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 74027 .summary, .result 67214 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 74027 .summary)
      LeftBound74026.bound (LeftBound74026.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29159⟩⟩) (rawTerms := some (Proof.Events289.exact74027RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound74026.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67214 .summary)
      LeftBound67213.bound (LeftBound67213.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29375⟩⟩) (rawTerms := some (Proof.Events262.exact67214RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67213.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74026.bound, LeftBound67213.bound]
def bound : CoeffClass := .finite ⟨19381531389106758144000, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74026.bound, LeftBound67213.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound74026.actual selector witness, LeftBound67213.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74031

namespace LeftBound74035
def owner : Owner := ⟨.program ⟨214⟩, ⟨29593⟩⟩
def transferEvent : Nat := 74035
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 74033 .coefficient, .predecessor 1 74034 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 74033 .coefficient)
      LeftBound74030.bound (LeftBound74030.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events289.exact74032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74030.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74030.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 74034 .coefficient)
      LeftBound66728.bound (LeftBound66728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66728.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66728.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74030.bound, LeftBound66728.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74030.bound, LeftBound66728.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound74030.actual selector witness, LeftBound66728.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74035

namespace LeftBound74036
def owner : Owner := ⟨.program ⟨214⟩, ⟨29593⟩⟩
def transferEvent : Nat := 74036
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 74032 .summary, .result 66732 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 74032 .summary)
      LeftBound74031.bound (LeftBound74031.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29376⟩⟩) (rawTerms := some (Proof.Events289.exact74032RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound74031.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66732 .summary)
      LeftBound66731.bound (LeftBound66731.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29592⟩⟩) (rawTerms := some (Proof.Events260.exact66732RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66731.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74031.bound, LeftBound66731.bound]
def bound : CoeffClass := .finite ⟨20673980874611694436352, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74031.bound, LeftBound66731.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound74031.actual selector witness, LeftBound66731.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74036

namespace LeftBound74040
def owner : Owner := ⟨.program ⟨214⟩, ⟨29810⟩⟩
def transferEvent : Nat := 74040
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 74038 .coefficient, .predecessor 1 74039 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 74038 .coefficient)
      LeftBound74035.bound (LeftBound74035.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events289.exact74037RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74035.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74035.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 74039 .coefficient)
      LeftBound66246.bound (LeftBound66246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66246.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74035.bound, LeftBound66246.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74035.bound, LeftBound66246.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound74035.actual selector witness, LeftBound66246.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74040

namespace LeftBound74041
def owner : Owner := ⟨.program ⟨214⟩, ⟨29810⟩⟩
def transferEvent : Nat := 74041
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 74037 .summary, .result 66250 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 74037 .summary)
      LeftBound74036.bound (LeftBound74036.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29593⟩⟩) (rawTerms := some (Proof.Events289.exact74037RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound74036.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66250 .summary)
      LeftBound66249.bound (LeftBound66249.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29809⟩⟩) (rawTerms := some (Proof.Events258.exact66250RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66249.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74036.bound, LeftBound66249.bound]
def bound : CoeffClass := .finite ⟨21966497597451692486656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74036.bound, LeftBound66249.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound74036.actual selector witness, LeftBound66249.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74041

namespace LeftBound74045
def owner : Owner := ⟨.program ⟨214⟩, ⟨30099⟩⟩
def transferEvent : Nat := 74045
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 74043 .coefficient, .predecessor 1 74044 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 74043 .coefficient)
      LeftBound74040.bound (LeftBound74040.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events289.exact74042RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74040.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74040.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 74044 .coefficient)
      LeftBound65764.bound (LeftBound65764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65768RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65764.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65764.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74040.bound, LeftBound65764.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74040.bound, LeftBound65764.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound74040.actual selector witness, LeftBound65764.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74045

namespace LeftBound74046
def owner : Owner := ⟨.program ⟨214⟩, ⟨30099⟩⟩
def transferEvent : Nat := 74046
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 74042 .summary, .result 65768 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 74042 .summary)
      LeftBound74041.bound (LeftBound74041.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29810⟩⟩) (rawTerms := some (Proof.Events289.exact74042RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound74041.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65768 .summary)
      LeftBound65767.bound (LeftBound65767.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30098⟩⟩) (rawTerms := some (Proof.Events256.exact65768RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65767.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74041.bound, LeftBound65767.bound]
def bound : CoeffClass := .finite ⟨23259036732736711122944, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74041.bound, LeftBound65767.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound74041.actual selector witness, LeftBound65767.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74046

namespace LeftBound74050
def owner : Owner := ⟨.program ⟨214⟩, ⟨30100⟩⟩
def transferEvent : Nat := 74050
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 74048 .coefficient) (.predecessor 1 74049 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 74048 .coefficient)
      LeftBound74045.bound (LeftBound74045.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events289.exact74047RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74045.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74045.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 74049 .coefficient)
      LeftAuthority65269.bound (LeftAuthority65269.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65270RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65269.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65269.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound74045.bound LeftAuthority65269.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74045.bound, LeftAuthority65269.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound74045.actual selector witness) * (LeftAuthority65269.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound74050

namespace LeftBound74051
def owner : Owner := ⟨.program ⟨214⟩, ⟨30100⟩⟩
def transferEvent : Nat := 74051
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩ [⟨.result 65270 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65270 .coefficient)
      LeftAuthority65269.bound (LeftAuthority65269.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18678⟩⟩) (rawTerms := some (Proof.Events254.exact65270RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65269.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65269.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority65269.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65269.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority65269.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound74051

namespace LeftBound74052
def owner : Owner := ⟨.program ⟨214⟩, ⟨30100⟩⟩
def transferEvent : Nat := 74052
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 74047 .summary) (.transfer 74051) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 74047 .summary)
      LeftBound74046.bound (LeftBound74046.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30099⟩⟩) (rawTerms := some (Proof.Events289.exact74047RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound74046.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 74051)
      LeftBound74051.bound (LeftBound74051.actual selector witness) := by
  exact .transfer (LeftBound74051.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound74046.bound LeftBound74051.bound
def bound : CoeffClass := .finite ⟨85361036953731453608582447104, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74046.bound, LeftBound74051.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound74046.actual selector witness) * (LeftBound74051.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound74052

namespace LeftBound74131
def owner : Owner := ⟨.program ⟨214⟩, ⟨18557⟩⟩
def transferEvent : Nat := 74131
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 74129 .coefficient) (.value (.predecessor 1 74130 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 74129 .coefficient)
      LeftAuthority74127.bound (LeftAuthority74127.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events289.exact74128RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority74127.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority74127.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 74130 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority74127.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority74127.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority74127.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound74131

namespace LeftBound74135
def owner : Owner := ⟨.program ⟨214⟩, ⟨18558⟩⟩
def transferEvent : Nat := 74135
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 74133 .coefficient) (.predecessor 1 74134 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 74133 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 74134 .coefficient)
      LeftBound74131.bound (LeftBound74131.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events289.exact74132RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74131.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74131.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound74131.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound74131.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound74131.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound74135

namespace LeftBound74136
def owner : Owner := ⟨.program ⟨214⟩, ⟨18558⟩⟩
def transferEvent : Nat := 74136
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩ [⟨.result 74128 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 74128 .coefficient)
      LeftAuthority74127.bound (LeftAuthority74127.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18555⟩⟩) (rawTerms := some (Proof.Events289.exact74128RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority74127.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority74127.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority74127.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority74127.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority74127.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound74136

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
