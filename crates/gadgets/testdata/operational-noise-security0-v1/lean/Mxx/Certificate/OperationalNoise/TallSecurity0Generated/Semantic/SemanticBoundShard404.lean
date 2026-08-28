import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard374
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard378
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard382
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard385
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard389
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard393
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard396
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard400
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard403

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound59336
def owner : Owner := ⟨.program ⟨214⟩, ⟨26373⟩⟩
def transferEvent : Nat := 59336
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 59330 .summary, .result 59152 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59330 .summary)
      LeftBound59164.bound (LeftBound59164.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20399⟩⟩) (rawTerms := some (Proof.Events231.exact59330RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59152 .summary)
      LeftBound59147.bound (LeftBound59147.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26372⟩⟩) (rawTerms := some (Proof.Events231.exact59152RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59147.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59164.bound, LeftBound59147.bound]
def bound : CoeffClass := .finite ⟨1291889174379421642752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59164.bound, LeftBound59147.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59164.actual selector witness, LeftBound59147.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59336

namespace LeftBound59340
def owner : Owner := ⟨.program ⟨214⟩, ⟨26581⟩⟩
def transferEvent : Nat := 59340
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 59338 .coefficient, .predecessor 1 59339 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59338 .coefficient)
      LeftBound59333.bound (LeftBound59333.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59337RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59333.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59333.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59339 .coefficient)
      LeftBound58851.bound (LeftBound58851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58855RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58851.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58851.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59333.bound, LeftBound58851.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59333.bound, LeftBound58851.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59333.actual selector witness, LeftBound58851.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59340

namespace LeftBound59341
def owner : Owner := ⟨.program ⟨214⟩, ⟨26581⟩⟩
def transferEvent : Nat := 59341
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 59337 .summary, .result 58855 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59337 .summary)
      LeftBound59336.bound (LeftBound59336.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26373⟩⟩) (rawTerms := some (Proof.Events231.exact59337RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59336.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58855 .summary)
      LeftBound58854.bound (LeftBound58854.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26580⟩⟩) (rawTerms := some (Proof.Events229.exact58855RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58854.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59336.bound, LeftBound58854.bound]
def bound : CoeffClass := .finite ⟨2583789554981353578496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59336.bound, LeftBound58854.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59336.actual selector witness, LeftBound58854.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59341

namespace LeftBound59345
def owner : Owner := ⟨.program ⟨214⟩, ⟨26798⟩⟩
def transferEvent : Nat := 59345
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 59343 .coefficient, .predecessor 1 59344 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59343 .coefficient)
      LeftBound59340.bound (LeftBound59340.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59342RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59340.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59340.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59344 .coefficient)
      LeftBound58369.bound (LeftBound58369.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events228.exact58373RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58369.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58369.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59340.bound, LeftBound58369.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59340.bound, LeftBound58369.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59340.actual selector witness, LeftBound58369.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59345

namespace LeftBound59346
def owner : Owner := ⟨.program ⟨214⟩, ⟨26798⟩⟩
def transferEvent : Nat := 59346
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 59342 .summary, .result 58373 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59342 .summary)
      LeftBound59341.bound (LeftBound59341.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26581⟩⟩) (rawTerms := some (Proof.Events231.exact59342RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59341.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58373 .summary)
      LeftBound58372.bound (LeftBound58372.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26797⟩⟩) (rawTerms := some (Proof.Events228.exact58373RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58372.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59341.bound, LeftBound58372.bound]
def bound : CoeffClass := .finite ⟨3875701141805795807232, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59341.bound, LeftBound58372.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59341.actual selector witness, LeftBound58372.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59346

namespace LeftBound59350
def owner : Owner := ⟨.program ⟨214⟩, ⟨27015⟩⟩
def transferEvent : Nat := 59350
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 59348 .coefficient, .predecessor 1 59349 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59348 .coefficient)
      LeftBound59345.bound (LeftBound59345.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59347RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59345.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59345.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59349 .coefficient)
      LeftBound57887.bound (LeftBound57887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events226.exact57891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57887.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57887.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59345.bound, LeftBound57887.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59345.bound, LeftBound57887.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59345.actual selector witness, LeftBound57887.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59350

namespace LeftBound59351
def owner : Owner := ⟨.program ⟨214⟩, ⟨27015⟩⟩
def transferEvent : Nat := 59351
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 59347 .summary, .result 57891 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59347 .summary)
      LeftBound59346.bound (LeftBound59346.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26798⟩⟩) (rawTerms := some (Proof.Events231.exact59347RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59346.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57891 .summary)
      LeftBound57890.bound (LeftBound57890.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27014⟩⟩) (rawTerms := some (Proof.Events226.exact57891RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57890.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59346.bound, LeftBound57890.bound]
def bound : CoeffClass := .finite ⟨5167635141075258621952, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59346.bound, LeftBound57890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59346.actual selector witness, LeftBound57890.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59351

namespace LeftBound59355
def owner : Owner := ⟨.program ⟨214⟩, ⟨27232⟩⟩
def transferEvent : Nat := 59355
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 59353 .coefficient, .predecessor 1 59354 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59353 .coefficient)
      LeftBound59350.bound (LeftBound59350.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59352RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59350.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59350.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59354 .coefficient)
      LeftBound57405.bound (LeftBound57405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events224.exact57409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound57405.bound, RecordedBoundRefines] <;> decide)
      (LeftBound57405.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59350.bound, LeftBound57405.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59350.bound, LeftBound57405.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59350.actual selector witness, LeftBound57405.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59355

namespace LeftBound59356
def owner : Owner := ⟨.program ⟨214⟩, ⟨27232⟩⟩
def transferEvent : Nat := 59356
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 59352 .summary, .result 57409 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59352 .summary)
      LeftBound59351.bound (LeftBound59351.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27015⟩⟩) (rawTerms := some (Proof.Events231.exact59352RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59351.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 57409 .summary)
      LeftBound57408.bound (LeftBound57408.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27231⟩⟩) (rawTerms := some (Proof.Events224.exact57409RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound57408.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59351.bound, LeftBound57408.bound]
def bound : CoeffClass := .finite ⟨6459613965234762608640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59351.bound, LeftBound57408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59351.actual selector witness, LeftBound57408.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59356

namespace LeftBound59360
def owner : Owner := ⟨.program ⟨214⟩, ⟨27449⟩⟩
def transferEvent : Nat := 59360
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 59358 .coefficient, .predecessor 1 59359 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59358 .coefficient)
      LeftBound59355.bound (LeftBound59355.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59357RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59355.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59355.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59359 .coefficient)
      LeftBound56923.bound (LeftBound56923.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact56927RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56923.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56923.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59355.bound, LeftBound56923.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59355.bound, LeftBound56923.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59355.actual selector witness, LeftBound56923.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59360

namespace LeftBound59361
def owner : Owner := ⟨.program ⟨214⟩, ⟨27449⟩⟩
def transferEvent : Nat := 59361
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 59357 .summary, .result 56927 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59357 .summary)
      LeftBound59356.bound (LeftBound59356.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27232⟩⟩) (rawTerms := some (Proof.Events231.exact59357RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59356.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56927 .summary)
      LeftBound56926.bound (LeftBound56926.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27448⟩⟩) (rawTerms := some (Proof.Events222.exact56927RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56926.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59356.bound, LeftBound56926.bound]
def bound : CoeffClass := .finite ⟨7751615201839287181312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59356.bound, LeftBound56926.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59356.actual selector witness, LeftBound56926.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59361

namespace LeftBound59365
def owner : Owner := ⟨.program ⟨214⟩, ⟨27666⟩⟩
def transferEvent : Nat := 59365
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 59363 .coefficient, .predecessor 1 59364 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59363 .coefficient)
      LeftBound59360.bound (LeftBound59360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59362RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59360.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59360.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59364 .coefficient)
      LeftBound56441.bound (LeftBound56441.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56445RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56441.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56441.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59360.bound, LeftBound56441.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59360.bound, LeftBound56441.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59360.actual selector witness, LeftBound56441.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59365

namespace LeftBound59366
def owner : Owner := ⟨.program ⟨214⟩, ⟨27666⟩⟩
def transferEvent : Nat := 59366
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 59362 .summary, .result 56445 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59362 .summary)
      LeftBound59361.bound (LeftBound59361.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27449⟩⟩) (rawTerms := some (Proof.Events231.exact59362RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59361.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56445 .summary)
      LeftBound56444.bound (LeftBound56444.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27665⟩⟩) (rawTerms := some (Proof.Events220.exact56445RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56444.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59361.bound, LeftBound56444.bound]
def bound : CoeffClass := .finite ⟨9043661263333852925952, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59361.bound, LeftBound56444.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59361.actual selector witness, LeftBound56444.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59366

namespace LeftBound59370
def owner : Owner := ⟨.program ⟨214⟩, ⟨27883⟩⟩
def transferEvent : Nat := 59370
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 59368 .coefficient, .predecessor 1 59369 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59368 .coefficient)
      LeftBound59365.bound (LeftBound59365.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59367RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59365.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59365.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59369 .coefficient)
      LeftBound55959.bound (LeftBound55959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact55963RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55959.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55959.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59365.bound, LeftBound55959.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59365.bound, LeftBound55959.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59365.actual selector witness, LeftBound55959.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59370

namespace LeftBound59371
def owner : Owner := ⟨.program ⟨214⟩, ⟨27883⟩⟩
def transferEvent : Nat := 59371
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 59367 .summary, .result 55963 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59367 .summary)
      LeftBound59366.bound (LeftBound59366.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27666⟩⟩) (rawTerms := some (Proof.Events231.exact59367RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59366.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55963 .summary)
      LeftBound55962.bound (LeftBound55962.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27882⟩⟩) (rawTerms := some (Proof.Events218.exact55963RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55962.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59366.bound, LeftBound55962.bound]
def bound : CoeffClass := .finite ⟨10335729737273439256576, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59366.bound, LeftBound55962.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59366.actual selector witness, LeftBound55962.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59371

namespace LeftBound59375
def owner : Owner := ⟨.program ⟨214⟩, ⟨28100⟩⟩
def transferEvent : Nat := 59375
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 59373 .coefficient, .predecessor 1 59374 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59373 .coefficient)
      LeftBound59370.bound (LeftBound59370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59372RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59370.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59370.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59374 .coefficient)
      LeftBound55477.bound (LeftBound55477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55481RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55477.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59370.bound, LeftBound55477.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59370.bound, LeftBound55477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59370.actual selector witness, LeftBound55477.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59375

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
