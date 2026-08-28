import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard207
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard209
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard210
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard211
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard213
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard214
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard215
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard217
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard218
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard231

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound35609
def owner : Owner := ⟨.program ⟨214⟩, ⟨28337⟩⟩
def transferEvent : Nat := 35609
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35605 .summary, .result 33625 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35605 .summary)
      LeftBound35604.bound (LeftBound35604.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28120⟩⟩) (rawTerms := some (Proof.Events139.exact35605RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35604.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 33625 .summary)
      LeftBound33620.bound (LeftBound33620.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28336⟩⟩) (rawTerms := some (Proof.Events131.exact33625RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound33620.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35604.bound, LeftBound33620.bound]
def bound : CoeffClass := .finite ⟨47416693230599820876439355444, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35604.bound, LeftBound33620.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35604.actual selector witness, LeftBound33620.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35609

namespace LeftBound35613
def owner : Owner := ⟨.program ⟨214⟩, ⟨28554⟩⟩
def transferEvent : Nat := 35613
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35611 .coefficient, .predecessor 1 35612 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35611 .coefficient)
      LeftBound35608.bound (LeftBound35608.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35610RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35608.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35608.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35612 .coefficient)
      LeftBound33406.bound (LeftBound33406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events130.exact33413RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33406.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33406.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35608.bound, LeftBound33406.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35608.bound, LeftBound33406.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35608.actual selector witness, LeftBound33406.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35613

namespace LeftBound35614
def owner : Owner := ⟨.program ⟨214⟩, ⟨28554⟩⟩
def transferEvent : Nat := 35614
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35610 .summary, .result 33413 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35610 .summary)
      LeftBound35609.bound (LeftBound35609.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28337⟩⟩) (rawTerms := some (Proof.Events139.exact35610RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35609.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 33413 .summary)
      LeftBound33408.bound (LeftBound33408.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28553⟩⟩) (rawTerms := some (Proof.Events130.exact33413RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound33408.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35609.bound, LeftBound33408.bound]
def bound : CoeffClass := .finite ⟨52159098727244633768554659892, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35609.bound, LeftBound33408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35609.actual selector witness, LeftBound33408.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35614

namespace LeftBound35618
def owner : Owner := ⟨.program ⟨214⟩, ⟨28771⟩⟩
def transferEvent : Nat := 35618
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35616 .coefficient, .predecessor 1 35617 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35616 .coefficient)
      LeftBound35613.bound (LeftBound35613.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35615RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35613.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35613.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35617 .coefficient)
      LeftBound33194.bound (LeftBound33194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events129.exact33201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33194.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33194.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35613.bound, LeftBound33194.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35613.bound, LeftBound33194.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35613.actual selector witness, LeftBound33194.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35618

namespace LeftBound35619
def owner : Owner := ⟨.program ⟨214⟩, ⟨28771⟩⟩
def transferEvent : Nat := 35619
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35615 .summary, .result 33201 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35615 .summary)
      LeftBound35614.bound (LeftBound35614.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28554⟩⟩) (rawTerms := some (Proof.Events139.exact35615RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35614.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 33201 .summary)
      LeftBound33196.bound (LeftBound33196.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28770⟩⟩) (rawTerms := some (Proof.Events129.exact33201RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound33196.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35614.bound, LeftBound33196.bound]
def bound : CoeffClass := .finite ⟨56901750985984920673341931572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35614.bound, LeftBound33196.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35614.actual selector witness, LeftBound33196.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35619

namespace LeftBound35623
def owner : Owner := ⟨.program ⟨214⟩, ⟨28988⟩⟩
def transferEvent : Nat := 35623
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35621 .coefficient, .predecessor 1 35622 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35621 .coefficient)
      LeftBound35618.bound (LeftBound35618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35618.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35622 .coefficient)
      LeftBound32982.bound (LeftBound32982.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32989RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32982.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32982.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35618.bound, LeftBound32982.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35618.bound, LeftBound32982.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35618.actual selector witness, LeftBound32982.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35623

namespace LeftBound35624
def owner : Owner := ⟨.program ⟨214⟩, ⟨28988⟩⟩
def transferEvent : Nat := 35624
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35620 .summary, .result 32989 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35620 .summary)
      LeftBound35619.bound (LeftBound35619.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28771⟩⟩) (rawTerms := some (Proof.Events139.exact35620RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35619.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 32989 .summary)
      LeftBound32984.bound (LeftBound32984.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28987⟩⟩) (rawTerms := some (Proof.Events128.exact32989RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32984.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35619.bound, LeftBound32984.bound]
def bound : CoeffClass := .finite ⟨61644567752788856919910514740, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35619.bound, LeftBound32984.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35619.actual selector witness, LeftBound32984.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35624

namespace LeftBound35628
def owner : Owner := ⟨.program ⟨214⟩, ⟨29205⟩⟩
def transferEvent : Nat := 35628
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35626 .coefficient, .predecessor 1 35627 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35626 .coefficient)
      LeftBound35623.bound (LeftBound35623.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35623.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35623.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35627 .coefficient)
      LeftBound32770.bound (LeftBound32770.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events128.exact32777RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32770.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32770.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35623.bound, LeftBound32770.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35623.bound, LeftBound32770.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35623.actual selector witness, LeftBound32770.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35628

namespace LeftBound35629
def owner : Owner := ⟨.program ⟨214⟩, ⟨29205⟩⟩
def transferEvent : Nat := 35629
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35625 .summary, .result 32777 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35625 .summary)
      LeftBound35624.bound (LeftBound35624.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28988⟩⟩) (rawTerms := some (Proof.Events139.exact35625RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35624.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 32777 .summary)
      LeftBound32772.bound (LeftBound32772.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29204⟩⟩) (rawTerms := some (Proof.Events128.exact32777RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32772.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35624.bound, LeftBound32772.bound]
def bound : CoeffClass := .finite ⟨66387466773624617837369753652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35624.bound, LeftBound32772.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35624.actual selector witness, LeftBound32772.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35629

namespace LeftBound35633
def owner : Owner := ⟨.program ⟨214⟩, ⟨29422⟩⟩
def transferEvent : Nat := 35633
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35631 .coefficient, .predecessor 1 35632 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35631 .coefficient)
      LeftBound35628.bound (LeftBound35628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35630RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35628.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35628.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35632 .coefficient)
      LeftBound32558.bound (LeftBound32558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events127.exact32565RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32558.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35628.bound, LeftBound32558.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35628.bound, LeftBound32558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35628.actual selector witness, LeftBound32558.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35633

namespace LeftBound35634
def owner : Owner := ⟨.program ⟨214⟩, ⟨29422⟩⟩
def transferEvent : Nat := 35634
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35630 .summary, .result 32565 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35630 .summary)
      LeftBound35629.bound (LeftBound35629.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29205⟩⟩) (rawTerms := some (Proof.Events139.exact35630RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35629.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 32565 .summary)
      LeftBound32560.bound (LeftBound32560.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29421⟩⟩) (rawTerms := some (Proof.Events127.exact32565RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32560.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35629.bound, LeftBound32560.bound]
def bound : CoeffClass := .finite ⟨71130530302524028096610304052, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35629.bound, LeftBound32560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35629.actual selector witness, LeftBound32560.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35634

namespace LeftBound35638
def owner : Owner := ⟨.program ⟨214⟩, ⟨29639⟩⟩
def transferEvent : Nat := 35638
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35636 .coefficient, .predecessor 1 35637 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35636 .coefficient)
      LeftBound35633.bound (LeftBound35633.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35633.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35633.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35637 .coefficient)
      LeftBound32346.bound (LeftBound32346.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events126.exact32353RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32346.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32346.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35633.bound, LeftBound32346.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35633.bound, LeftBound32346.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35633.actual selector witness, LeftBound32346.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35638

namespace LeftBound35639
def owner : Owner := ⟨.program ⟨214⟩, ⟨29639⟩⟩
def transferEvent : Nat := 35639
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35635 .summary, .result 32353 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35635 .summary)
      LeftBound35634.bound (LeftBound35634.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29422⟩⟩) (rawTerms := some (Proof.Events139.exact35635RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35634.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 32353 .summary)
      LeftBound32348.bound (LeftBound32348.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29638⟩⟩) (rawTerms := some (Proof.Events126.exact32353RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32348.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35634.bound, LeftBound32348.bound]
def bound : CoeffClass := .finite ⟨75873840593518912368522821684, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35634.bound, LeftBound32348.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35634.actual selector witness, LeftBound32348.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35639

namespace LeftBound35643
def owner : Owner := ⟨.program ⟨214⟩, ⟨29856⟩⟩
def transferEvent : Nat := 35643
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35641 .coefficient, .predecessor 1 35642 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35641 .coefficient)
      LeftBound35638.bound (LeftBound35638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35640RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35638.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35638.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35642 .coefficient)
      LeftBound32134.bound (LeftBound32134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32141RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32134.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35638.bound, LeftBound32134.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35638.bound, LeftBound32134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35638.actual selector witness, LeftBound32134.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35643

namespace LeftBound35644
def owner : Owner := ⟨.program ⟨214⟩, ⟨29856⟩⟩
def transferEvent : Nat := 35644
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35640 .summary, .result 32141 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35640 .summary)
      LeftBound35639.bound (LeftBound35639.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29639⟩⟩) (rawTerms := some (Proof.Events139.exact35640RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35639.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 32141 .summary)
      LeftBound32136.bound (LeftBound32136.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29855⟩⟩) (rawTerms := some (Proof.Events125.exact32141RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32136.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35639.bound, LeftBound32136.bound]
def bound : CoeffClass := .finite ⟨80617397646609270653107306548, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35639.bound, LeftBound32136.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35639.actual selector witness, LeftBound32136.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35644

namespace LeftBound35648
def owner : Owner := ⟨.program ⟨214⟩, ⟨30181⟩⟩
def transferEvent : Nat := 35648
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35646 .coefficient, .predecessor 1 35647 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35646 .coefficient)
      LeftBound35643.bound (LeftBound35643.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35645RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35643.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35643.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35647 .coefficient)
      LeftBound31922.bound (LeftBound31922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31922.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31922.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35643.bound, LeftBound31922.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35643.bound, LeftBound31922.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35643.actual selector witness, LeftBound31922.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35648

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
