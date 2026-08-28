import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard714
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard716
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard717
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard718
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard720
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard721
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard722
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard723
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard724
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard725
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard735

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound107229
def owner : Owner := ⟨.program ⟨214⟩, ⟨27829⟩⟩
def transferEvent : Nat := 107229
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107225 .summary, .result 105847 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107225 .summary)
      LeftBound107224.bound (LeftBound107224.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27612⟩⟩) (rawTerms := some (Proof.Events418.exact107225RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107224.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105847 .summary)
      LeftBound105842.bound (LeftBound105842.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27828⟩⟩) (rawTerms := some (Proof.Events413.exact105847RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105842.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107224.bound, LeftBound105842.bound]
def bound : CoeffClass := .finite ⟨37932293507469318446662025268, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107224.bound, LeftBound105842.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107224.actual selector witness, LeftBound105842.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107229

namespace LeftBound107233
def owner : Owner := ⟨.program ⟨214⟩, ⟨28046⟩⟩
def transferEvent : Nat := 107233
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107231 .coefficient, .predecessor 1 107232 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107231 .coefficient)
      LeftBound107228.bound (LeftBound107228.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107230RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107228.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107228.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107232 .coefficient)
      LeftBound105652.bound (LeftBound105652.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105652.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105652.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107228.bound, LeftBound105652.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107228.bound, LeftBound105652.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107228.actual selector witness, LeftBound105652.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107233

namespace LeftBound107234
def owner : Owner := ⟨.program ⟨214⟩, ⟨28046⟩⟩
def transferEvent : Nat := 107234
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107230 .summary, .result 105659 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107230 .summary)
      LeftBound107229.bound (LeftBound107229.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27829⟩⟩) (rawTerms := some (Proof.Events418.exact107230RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107229.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105659 .summary)
      LeftBound105654.bound (LeftBound105654.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28045⟩⟩) (rawTerms := some (Proof.Events412.exact105659RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105654.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107229.bound, LeftBound105654.bound]
def bound : CoeffClass := .finite ⟨42674369987986832655214706740, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107229.bound, LeftBound105654.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107229.actual selector witness, LeftBound105654.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107234

namespace LeftBound107238
def owner : Owner := ⟨.program ⟨214⟩, ⟨28263⟩⟩
def transferEvent : Nat := 107238
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107236 .coefficient, .predecessor 1 107237 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107236 .coefficient)
      LeftBound107233.bound (LeftBound107233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107235RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107233.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107233.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107237 .coefficient)
      LeftBound105464.bound (LeftBound105464.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105471RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105464.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105464.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107233.bound, LeftBound105464.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107233.bound, LeftBound105464.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107233.actual selector witness, LeftBound105464.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107238

namespace LeftBound107239
def owner : Owner := ⟨.program ⟨214⟩, ⟨28263⟩⟩
def transferEvent : Nat := 107239
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107235 .summary, .result 105471 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107235 .summary)
      LeftBound107234.bound (LeftBound107234.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28046⟩⟩) (rawTerms := some (Proof.Events418.exact107235RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107234.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105471 .summary)
      LeftBound105466.bound (LeftBound105466.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28262⟩⟩) (rawTerms := some (Proof.Events411.exact105471RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105466.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107234.bound, LeftBound105466.bound]
def bound : CoeffClass := .finite ⟨47416693230599820876439355444, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107234.bound, LeftBound105466.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107234.actual selector witness, LeftBound105466.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107239

namespace LeftBound107243
def owner : Owner := ⟨.program ⟨214⟩, ⟨28480⟩⟩
def transferEvent : Nat := 107243
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107241 .coefficient, .predecessor 1 107242 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107241 .coefficient)
      LeftBound107238.bound (LeftBound107238.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107238.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107238.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107242 .coefficient)
      LeftBound105276.bound (LeftBound105276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105276.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105276.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107238.bound, LeftBound105276.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107238.bound, LeftBound105276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107238.actual selector witness, LeftBound105276.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107243

namespace LeftBound107244
def owner : Owner := ⟨.program ⟨214⟩, ⟨28480⟩⟩
def transferEvent : Nat := 107244
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107240 .summary, .result 105283 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107240 .summary)
      LeftBound107239.bound (LeftBound107239.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28263⟩⟩) (rawTerms := some (Proof.Events418.exact107240RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107239.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105283 .summary)
      LeftBound105278.bound (LeftBound105278.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28479⟩⟩) (rawTerms := some (Proof.Events411.exact105283RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105278.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107239.bound, LeftBound105278.bound]
def bound : CoeffClass := .finite ⟨52159098727244633768554659892, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107239.bound, LeftBound105278.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107239.actual selector witness, LeftBound105278.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107244

namespace LeftBound107248
def owner : Owner := ⟨.program ⟨214⟩, ⟨28697⟩⟩
def transferEvent : Nat := 107248
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107246 .coefficient, .predecessor 1 107247 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107246 .coefficient)
      LeftBound107243.bound (LeftBound107243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107243.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107247 .coefficient)
      LeftBound105088.bound (LeftBound105088.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events410.exact105095RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105088.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105088.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107243.bound, LeftBound105088.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107243.bound, LeftBound105088.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107243.actual selector witness, LeftBound105088.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107248

namespace LeftBound107249
def owner : Owner := ⟨.program ⟨214⟩, ⟨28697⟩⟩
def transferEvent : Nat := 107249
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107245 .summary, .result 105095 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107245 .summary)
      LeftBound107244.bound (LeftBound107244.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28480⟩⟩) (rawTerms := some (Proof.Events418.exact107245RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107244.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105095 .summary)
      LeftBound105090.bound (LeftBound105090.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28696⟩⟩) (rawTerms := some (Proof.Events410.exact105095RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105090.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107244.bound, LeftBound105090.bound]
def bound : CoeffClass := .finite ⟨56901750985984920673341931572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107244.bound, LeftBound105090.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107244.actual selector witness, LeftBound105090.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107249

namespace LeftBound107253
def owner : Owner := ⟨.program ⟨214⟩, ⟨28914⟩⟩
def transferEvent : Nat := 107253
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107251 .coefficient, .predecessor 1 107252 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107251 .coefficient)
      LeftBound107248.bound (LeftBound107248.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107248.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107248.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107252 .coefficient)
      LeftBound104900.bound (LeftBound104900.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104900.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104900.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107248.bound, LeftBound104900.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107248.bound, LeftBound104900.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107248.actual selector witness, LeftBound104900.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107253

namespace LeftBound107254
def owner : Owner := ⟨.program ⟨214⟩, ⟨28914⟩⟩
def transferEvent : Nat := 107254
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107250 .summary, .result 104907 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107250 .summary)
      LeftBound107249.bound (LeftBound107249.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28697⟩⟩) (rawTerms := some (Proof.Events418.exact107250RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107249.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104907 .summary)
      LeftBound104902.bound (LeftBound104902.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28913⟩⟩) (rawTerms := some (Proof.Events409.exact104907RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104902.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107249.bound, LeftBound104902.bound]
def bound : CoeffClass := .finite ⟨61644567752788856919910514740, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107249.bound, LeftBound104902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107249.actual selector witness, LeftBound104902.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107254

namespace LeftBound107258
def owner : Owner := ⟨.program ⟨214⟩, ⟨29131⟩⟩
def transferEvent : Nat := 107258
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107256 .coefficient, .predecessor 1 107257 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107256 .coefficient)
      LeftBound107253.bound (LeftBound107253.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107255RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107253.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107253.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107257 .coefficient)
      LeftBound104712.bound (LeftBound104712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104712.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104712.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107253.bound, LeftBound104712.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107253.bound, LeftBound104712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107253.actual selector witness, LeftBound104712.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107258

namespace LeftBound107259
def owner : Owner := ⟨.program ⟨214⟩, ⟨29131⟩⟩
def transferEvent : Nat := 107259
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107255 .summary, .result 104719 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107255 .summary)
      LeftBound107254.bound (LeftBound107254.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28914⟩⟩) (rawTerms := some (Proof.Events418.exact107255RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107254.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104719 .summary)
      LeftBound104714.bound (LeftBound104714.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29130⟩⟩) (rawTerms := some (Proof.Events409.exact104719RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104714.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107254.bound, LeftBound104714.bound]
def bound : CoeffClass := .finite ⟨66387466773624617837369753652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107254.bound, LeftBound104714.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107254.actual selector witness, LeftBound104714.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107259

namespace LeftBound107263
def owner : Owner := ⟨.program ⟨214⟩, ⟨29348⟩⟩
def transferEvent : Nat := 107263
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107261 .coefficient, .predecessor 1 107262 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107261 .coefficient)
      LeftBound107258.bound (LeftBound107258.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107258.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107258.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107262 .coefficient)
      LeftBound104524.bound (LeftBound104524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104531RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104524.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104524.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107258.bound, LeftBound104524.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107258.bound, LeftBound104524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107258.actual selector witness, LeftBound104524.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107263

namespace LeftBound107264
def owner : Owner := ⟨.program ⟨214⟩, ⟨29348⟩⟩
def transferEvent : Nat := 107264
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107260 .summary, .result 104531 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107260 .summary)
      LeftBound107259.bound (LeftBound107259.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29131⟩⟩) (rawTerms := some (Proof.Events418.exact107260RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107259.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104531 .summary)
      LeftBound104526.bound (LeftBound104526.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29347⟩⟩) (rawTerms := some (Proof.Events408.exact104531RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104526.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107259.bound, LeftBound104526.bound]
def bound : CoeffClass := .finite ⟨71130530302524028096610304052, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107259.bound, LeftBound104526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107259.actual selector witness, LeftBound104526.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107264

namespace LeftBound107268
def owner : Owner := ⟨.program ⟨214⟩, ⟨29565⟩⟩
def transferEvent : Nat := 107268
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107266 .coefficient, .predecessor 1 107267 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107266 .coefficient)
      LeftBound107263.bound (LeftBound107263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events419.exact107265RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107263.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107263.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107267 .coefficient)
      LeftBound104336.bound (LeftBound104336.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104336.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104336.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107263.bound, LeftBound104336.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107263.bound, LeftBound104336.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107263.actual selector witness, LeftBound104336.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107268

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
