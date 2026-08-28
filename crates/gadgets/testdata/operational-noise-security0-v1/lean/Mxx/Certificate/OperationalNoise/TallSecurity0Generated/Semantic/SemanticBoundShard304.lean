import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard240
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard244
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard248
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard251
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard255
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard259
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard303

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound44770
def owner : Owner := ⟨.program ⟨214⟩, ⟨28981⟩⟩
def transferEvent : Nat := 44770
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44768 .coefficient, .predecessor 1 44769 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44768 .coefficient)
      LeftBound44765.bound (LeftBound44765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44765.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44765.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44769 .coefficient)
      LeftBound38924.bound (LeftBound38924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38924.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38924.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44765.bound, LeftBound38924.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44765.bound, LeftBound38924.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44765.actual selector witness, LeftBound38924.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44770

namespace LeftBound44771
def owner : Owner := ⟨.program ⟨214⟩, ⟨28981⟩⟩
def transferEvent : Nat := 44771
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44767 .summary, .result 38928 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44767 .summary)
      LeftBound44766.bound (LeftBound44766.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28764⟩⟩) (rawTerms := some (Proof.Events174.exact44767RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44766.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38928 .summary)
      LeftBound38927.bound (LeftBound38927.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28980⟩⟩) (rawTerms := some (Proof.Events152.exact38928RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38927.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44766.bound, LeftBound38927.bound]
def bound : CoeffClass := .finite ⟨16796811717657050247168, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44766.bound, LeftBound38927.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44766.actual selector witness, LeftBound38927.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44771

namespace LeftBound44775
def owner : Owner := ⟨.program ⟨214⟩, ⟨29198⟩⟩
def transferEvent : Nat := 44775
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44773 .coefficient, .predecessor 1 44774 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44773 .coefficient)
      LeftBound44770.bound (LeftBound44770.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44772RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44770.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44770.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44774 .coefficient)
      LeftBound38442.bound (LeftBound38442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events150.exact38446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38442.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38442.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44770.bound, LeftBound38442.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44770.bound, LeftBound38442.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44770.actual selector witness, LeftBound38442.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44775

namespace LeftBound44776
def owner : Owner := ⟨.program ⟨214⟩, ⟨29198⟩⟩
def transferEvent : Nat := 44776
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44772 .summary, .result 38446 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44772 .summary)
      LeftBound44771.bound (LeftBound44771.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28981⟩⟩) (rawTerms := some (Proof.Events174.exact44772RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44771.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38446 .summary)
      LeftBound38445.bound (LeftBound38445.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29197⟩⟩) (rawTerms := some (Proof.Events150.exact38446RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38445.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44771.bound, LeftBound38445.bound]
def bound : CoeffClass := .finite ⟨18089149140936883609600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44771.bound, LeftBound38445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44771.actual selector witness, LeftBound38445.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44776

namespace LeftBound44780
def owner : Owner := ⟨.program ⟨214⟩, ⟨29415⟩⟩
def transferEvent : Nat := 44780
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44778 .coefficient, .predecessor 1 44779 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44778 .coefficient)
      LeftBound44775.bound (LeftBound44775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44777RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44775.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44775.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44779 .coefficient)
      LeftBound37960.bound (LeftBound37960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events148.exact37964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37960.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44775.bound, LeftBound37960.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44775.bound, LeftBound37960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44775.actual selector witness, LeftBound37960.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44780

namespace LeftBound44781
def owner : Owner := ⟨.program ⟨214⟩, ⟨29415⟩⟩
def transferEvent : Nat := 44781
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44777 .summary, .result 37964 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44777 .summary)
      LeftBound44776.bound (LeftBound44776.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29198⟩⟩) (rawTerms := some (Proof.Events174.exact44777RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44776.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37964 .summary)
      LeftBound37963.bound (LeftBound37963.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29414⟩⟩) (rawTerms := some (Proof.Events148.exact37964RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound37963.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44776.bound, LeftBound37963.bound]
def bound : CoeffClass := .finite ⟨19381531389106758144000, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44776.bound, LeftBound37963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44776.actual selector witness, LeftBound37963.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44781

namespace LeftBound44785
def owner : Owner := ⟨.program ⟨214⟩, ⟨29632⟩⟩
def transferEvent : Nat := 44785
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44783 .coefficient, .predecessor 1 44784 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44783 .coefficient)
      LeftBound44780.bound (LeftBound44780.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44782RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44780.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44780.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44784 .coefficient)
      LeftBound37478.bound (LeftBound37478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37478.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37478.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44780.bound, LeftBound37478.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44780.bound, LeftBound37478.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44780.actual selector witness, LeftBound37478.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44785

namespace LeftBound44786
def owner : Owner := ⟨.program ⟨214⟩, ⟨29632⟩⟩
def transferEvent : Nat := 44786
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44782 .summary, .result 37482 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44782 .summary)
      LeftBound44781.bound (LeftBound44781.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29415⟩⟩) (rawTerms := some (Proof.Events174.exact44782RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44781.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37482 .summary)
      LeftBound37481.bound (LeftBound37481.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29631⟩⟩) (rawTerms := some (Proof.Events146.exact37482RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound37481.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44781.bound, LeftBound37481.bound]
def bound : CoeffClass := .finite ⟨20673980874611694436352, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44781.bound, LeftBound37481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44781.actual selector witness, LeftBound37481.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44786

namespace LeftBound44790
def owner : Owner := ⟨.program ⟨214⟩, ⟨29849⟩⟩
def transferEvent : Nat := 44790
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44788 .coefficient, .predecessor 1 44789 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44788 .coefficient)
      LeftBound44785.bound (LeftBound44785.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44787RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44785.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44785.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44789 .coefficient)
      LeftBound36996.bound (LeftBound36996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact37000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36996.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36996.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44785.bound, LeftBound36996.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44785.bound, LeftBound36996.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44785.actual selector witness, LeftBound36996.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44790

namespace LeftBound44791
def owner : Owner := ⟨.program ⟨214⟩, ⟨29849⟩⟩
def transferEvent : Nat := 44791
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44787 .summary, .result 37000 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44787 .summary)
      LeftBound44786.bound (LeftBound44786.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29632⟩⟩) (rawTerms := some (Proof.Events174.exact44787RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44786.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37000 .summary)
      LeftBound36999.bound (LeftBound36999.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29848⟩⟩) (rawTerms := some (Proof.Events144.exact37000RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36999.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44786.bound, LeftBound36999.bound]
def bound : CoeffClass := .finite ⟨21966497597451692486656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44786.bound, LeftBound36999.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44786.actual selector witness, LeftBound36999.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44791

namespace LeftBound44795
def owner : Owner := ⟨.program ⟨214⟩, ⟨30165⟩⟩
def transferEvent : Nat := 44795
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44793 .coefficient, .predecessor 1 44794 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44793 .coefficient)
      LeftBound44790.bound (LeftBound44790.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44792RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44790.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44790.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44794 .coefficient)
      LeftBound36514.bound (LeftBound36514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36514.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44790.bound, LeftBound36514.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44790.bound, LeftBound36514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44790.actual selector witness, LeftBound36514.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44795

namespace LeftBound44796
def owner : Owner := ⟨.program ⟨214⟩, ⟨30165⟩⟩
def transferEvent : Nat := 44796
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44792 .summary, .result 36518 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44792 .summary)
      LeftBound44791.bound (LeftBound44791.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29849⟩⟩) (rawTerms := some (Proof.Events174.exact44792RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44791.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36518 .summary)
      LeftBound36517.bound (LeftBound36517.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30164⟩⟩) (rawTerms := some (Proof.Events142.exact36518RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36517.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44791.bound, LeftBound36517.bound]
def bound : CoeffClass := .finite ⟨23259036732736711122944, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44791.bound, LeftBound36517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44791.actual selector witness, LeftBound36517.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44796

namespace LeftBound44800
def owner : Owner := ⟨.program ⟨214⟩, ⟨30166⟩⟩
def transferEvent : Nat := 44800
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 44798 .coefficient) (.predecessor 1 44799 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44798 .coefficient)
      LeftBound44795.bound (LeftBound44795.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events174.exact44797RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44795.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44795.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44799 .coefficient)
      LeftAuthority36019.bound (LeftAuthority36019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36019.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36019.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound44795.bound LeftAuthority36019.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44795.bound, LeftAuthority36019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound44795.actual selector witness) * (LeftAuthority36019.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44800

namespace LeftBound44801
def owner : Owner := ⟨.program ⟨214⟩, ⟨30166⟩⟩
def transferEvent : Nat := 44801
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩ [⟨.result 36020 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36020 .coefficient)
      LeftAuthority36019.bound (LeftAuthority36019.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18687⟩⟩) (rawTerms := some (Proof.Events140.exact36020RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36019.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36019.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority36019.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority36019.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound44801

namespace LeftBound44802
def owner : Owner := ⟨.program ⟨214⟩, ⟨30166⟩⟩
def transferEvent : Nat := 44802
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 44797 .summary) (.transfer 44801) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44797 .summary)
      LeftBound44796.bound (LeftBound44796.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30165⟩⟩) (rawTerms := some (Proof.Events174.exact44797RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44796.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 44801)
      LeftBound44801.bound (LeftBound44801.actual selector witness) := by
  exact .transfer (LeftBound44801.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound44796.bound LeftBound44801.bound
def bound : CoeffClass := .finite ⟨85361036953731453608582447104, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44796.bound, LeftBound44801.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound44796.actual selector witness) * (LeftBound44801.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44802

namespace LeftBound44881
def owner : Owner := ⟨.program ⟨214⟩, ⟨18569⟩⟩
def transferEvent : Nat := 44881
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 44879 .coefficient) (.value (.predecessor 1 44880 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44879 .coefficient)
      LeftAuthority44877.bound (LeftAuthority44877.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events175.exact44878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44877.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44877.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44880 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority44877.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44877.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority44877.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound44881

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
