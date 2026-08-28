import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard658
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard661
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard665
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard669
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard672
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard676
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard679
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard680
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard683
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard687
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard705

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound102197
def owner : Owner := ⟨.program ⟨214⟩, ⟨27401⟩⟩
def transferEvent : Nat := 102197
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 102193 .summary, .result 100003 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102193 .summary)
      LeftBound102192.bound (LeftBound102192.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27184⟩⟩) (rawTerms := some (Proof.Events399.exact102193RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100003 .summary)
      LeftBound100002.bound (LeftBound100002.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27400⟩⟩) (rawTerms := some (Proof.Events390.exact100003RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100002.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102192.bound, LeftBound100002.bound]
def bound : CoeffClass := .finite ⟨7751615201839287181312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102192.bound, LeftBound100002.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102192.actual selector witness, LeftBound100002.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102197

namespace LeftBound102201
def owner : Owner := ⟨.program ⟨214⟩, ⟨27618⟩⟩
def transferEvent : Nat := 102201
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 102199 .coefficient, .predecessor 1 102200 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102199 .coefficient)
      LeftBound102196.bound (LeftBound102196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102198RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102196.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102196.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102200 .coefficient)
      LeftBound99565.bound (LeftBound99565.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99569RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99565.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99565.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102196.bound, LeftBound99565.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102196.bound, LeftBound99565.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102196.actual selector witness, LeftBound99565.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102201

namespace LeftBound102202
def owner : Owner := ⟨.program ⟨214⟩, ⟨27618⟩⟩
def transferEvent : Nat := 102202
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 102198 .summary, .result 99569 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102198 .summary)
      LeftBound102197.bound (LeftBound102197.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27401⟩⟩) (rawTerms := some (Proof.Events399.exact102198RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102197.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99569 .summary)
      LeftBound99568.bound (LeftBound99568.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27617⟩⟩) (rawTerms := some (Proof.Events388.exact99569RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99568.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102197.bound, LeftBound99568.bound]
def bound : CoeffClass := .finite ⟨9043661263333852925952, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102197.bound, LeftBound99568.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102197.actual selector witness, LeftBound99568.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102202

namespace LeftBound102206
def owner : Owner := ⟨.program ⟨214⟩, ⟨27835⟩⟩
def transferEvent : Nat := 102206
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 102204 .coefficient, .predecessor 1 102205 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102204 .coefficient)
      LeftBound102201.bound (LeftBound102201.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102203RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102201.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102201.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102205 .coefficient)
      LeftBound99131.bound (LeftBound99131.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99131.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99131.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102201.bound, LeftBound99131.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102201.bound, LeftBound99131.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102201.actual selector witness, LeftBound99131.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102206

namespace LeftBound102207
def owner : Owner := ⟨.program ⟨214⟩, ⟨27835⟩⟩
def transferEvent : Nat := 102207
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 102203 .summary, .result 99135 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102203 .summary)
      LeftBound102202.bound (LeftBound102202.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27618⟩⟩) (rawTerms := some (Proof.Events399.exact102203RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102202.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99135 .summary)
      LeftBound99134.bound (LeftBound99134.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27834⟩⟩) (rawTerms := some (Proof.Events387.exact99135RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99134.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102202.bound, LeftBound99134.bound]
def bound : CoeffClass := .finite ⟨10335729737273439256576, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102202.bound, LeftBound99134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102202.actual selector witness, LeftBound99134.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102207

namespace LeftBound102211
def owner : Owner := ⟨.program ⟨214⟩, ⟨28052⟩⟩
def transferEvent : Nat := 102211
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 102209 .coefficient, .predecessor 1 102210 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102209 .coefficient)
      LeftBound102206.bound (LeftBound102206.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102208RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102206.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102206.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102210 .coefficient)
      LeftBound98697.bound (LeftBound98697.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98701RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98697.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98697.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102206.bound, LeftBound98697.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102206.bound, LeftBound98697.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102206.actual selector witness, LeftBound98697.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102211

namespace LeftBound102212
def owner : Owner := ⟨.program ⟨214⟩, ⟨28052⟩⟩
def transferEvent : Nat := 102212
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 102208 .summary, .result 98701 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102208 .summary)
      LeftBound102207.bound (LeftBound102207.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27835⟩⟩) (rawTerms := some (Proof.Events399.exact102208RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102207.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98701 .summary)
      LeftBound98700.bound (LeftBound98700.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28051⟩⟩) (rawTerms := some (Proof.Events385.exact98701RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98700.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102207.bound, LeftBound98700.bound]
def bound : CoeffClass := .finite ⟨11627843036103066759168, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102207.bound, LeftBound98700.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102207.actual selector witness, LeftBound98700.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102212

namespace LeftBound102216
def owner : Owner := ⟨.program ⟨214⟩, ⟨28269⟩⟩
def transferEvent : Nat := 102216
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 102214 .coefficient, .predecessor 1 102215 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102214 .coefficient)
      LeftBound102211.bound (LeftBound102211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102211.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102215 .coefficient)
      LeftBound98263.bound (LeftBound98263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98263.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98263.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102211.bound, LeftBound98263.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102211.bound, LeftBound98263.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102211.actual selector witness, LeftBound98263.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102216

namespace LeftBound102217
def owner : Owner := ⟨.program ⟨214⟩, ⟨28269⟩⟩
def transferEvent : Nat := 102217
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 102213 .summary, .result 98267 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102213 .summary)
      LeftBound102212.bound (LeftBound102212.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28052⟩⟩) (rawTerms := some (Proof.Events399.exact102213RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102212.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98267 .summary)
      LeftBound98266.bound (LeftBound98266.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28268⟩⟩) (rawTerms := some (Proof.Events383.exact98267RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98266.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102212.bound, LeftBound98266.bound]
def bound : CoeffClass := .finite ⟨12920023572267756019712, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102212.bound, LeftBound98266.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102212.actual selector witness, LeftBound98266.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102217

namespace LeftBound102221
def owner : Owner := ⟨.program ⟨214⟩, ⟨28486⟩⟩
def transferEvent : Nat := 102221
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 102219 .coefficient, .predecessor 1 102220 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102219 .coefficient)
      LeftBound102216.bound (LeftBound102216.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102216.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102216.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102220 .coefficient)
      LeftBound97829.bound (LeftBound97829.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact97833RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97829.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97829.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102216.bound, LeftBound97829.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102216.bound, LeftBound97829.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102216.actual selector witness, LeftBound97829.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102221

namespace LeftBound102222
def owner : Owner := ⟨.program ⟨214⟩, ⟨28486⟩⟩
def transferEvent : Nat := 102222
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 102218 .summary, .result 97833 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102218 .summary)
      LeftBound102217.bound (LeftBound102217.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28269⟩⟩) (rawTerms := some (Proof.Events399.exact102218RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102217.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97833 .summary)
      LeftBound97832.bound (LeftBound97832.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28485⟩⟩) (rawTerms := some (Proof.Events382.exact97833RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97832.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102217.bound, LeftBound97832.bound]
def bound : CoeffClass := .finite ⟨14212226520877465866240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102217.bound, LeftBound97832.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102217.actual selector witness, LeftBound97832.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102222

namespace LeftBound102226
def owner : Owner := ⟨.program ⟨214⟩, ⟨28703⟩⟩
def transferEvent : Nat := 102226
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 102224 .coefficient, .predecessor 1 102225 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102224 .coefficient)
      LeftBound102221.bound (LeftBound102221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102221.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102221.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102225 .coefficient)
      LeftBound97395.bound (LeftBound97395.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97399RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97395.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97395.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102221.bound, LeftBound97395.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102221.bound, LeftBound97395.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102221.actual selector witness, LeftBound97395.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102226

namespace LeftBound102227
def owner : Owner := ⟨.program ⟨214⟩, ⟨28703⟩⟩
def transferEvent : Nat := 102227
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 102223 .summary, .result 97399 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102223 .summary)
      LeftBound102222.bound (LeftBound102222.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28486⟩⟩) (rawTerms := some (Proof.Events399.exact102223RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102222.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97399 .summary)
      LeftBound97398.bound (LeftBound97398.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28702⟩⟩) (rawTerms := some (Proof.Events380.exact97399RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97398.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102222.bound, LeftBound97398.bound]
def bound : CoeffClass := .finite ⟨15504496706822237470720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102222.bound, LeftBound97398.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102222.actual selector witness, LeftBound97398.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102227

namespace LeftBound102231
def owner : Owner := ⟨.program ⟨214⟩, ⟨28920⟩⟩
def transferEvent : Nat := 102231
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 102229 .coefficient, .predecessor 1 102230 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102229 .coefficient)
      LeftBound102226.bound (LeftBound102226.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102228RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102226.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102226.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102230 .coefficient)
      LeftBound96961.bound (LeftBound96961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events378.exact96965RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96961.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96961.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102226.bound, LeftBound96961.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102226.bound, LeftBound96961.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102226.actual selector witness, LeftBound96961.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102231

namespace LeftBound102232
def owner : Owner := ⟨.program ⟨214⟩, ⟨28920⟩⟩
def transferEvent : Nat := 102232
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 102228 .summary, .result 96965 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102228 .summary)
      LeftBound102227.bound (LeftBound102227.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28703⟩⟩) (rawTerms := some (Proof.Events399.exact102228RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102227.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96965 .summary)
      LeftBound96964.bound (LeftBound96964.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28919⟩⟩) (rawTerms := some (Proof.Events378.exact96965RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96964.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102227.bound, LeftBound96964.bound]
def bound : CoeffClass := .finite ⟨16796811717657050247168, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102227.bound, LeftBound96964.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102227.actual selector witness, LeftBound96964.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102232

namespace LeftBound102236
def owner : Owner := ⟨.program ⟨214⟩, ⟨29137⟩⟩
def transferEvent : Nat := 102236
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 102234 .coefficient, .predecessor 1 102235 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102234 .coefficient)
      LeftBound102231.bound (LeftBound102231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102233RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102231.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102231.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102235 .coefficient)
      LeftBound96527.bound (LeftBound96527.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96531RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96527.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96527.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102231.bound, LeftBound96527.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102231.bound, LeftBound96527.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102231.actual selector witness, LeftBound96527.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102236

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
