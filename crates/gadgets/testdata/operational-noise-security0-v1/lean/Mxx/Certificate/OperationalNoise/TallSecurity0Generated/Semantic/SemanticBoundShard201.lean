import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard175
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard179
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard182
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard183
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard186
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard190
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard193
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard197
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard200

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound30070
def owner : Owner := ⟨.program ⟨214⟩, ⟨26398⟩⟩
def transferEvent : Nat := 30070
def frameStart : Nat := 29970
def rule : BoundRule := .sum [.predecessor 0 30068 .coefficient, .predecessor 1 30069 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30068 .coefficient)
      LeftBound30066.bound (LeftBound30066.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30067RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30066.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30066.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30069 .coefficient)
      LeftBound30047.bound (LeftBound30047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30052RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30047.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30047.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30066.bound, LeftBound30047.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30066.bound, LeftBound30047.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30066.actual selector witness, LeftBound30047.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30070

namespace LeftBound30083
def owner : Owner := ⟨.program ⟨214⟩, ⟨26397⟩⟩
def transferEvent : Nat := 30083
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 30081 .coefficient, .predecessor 1 30082 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30081 .coefficient)
      LeftBound29912.bound (LeftBound29912.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29912.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29912.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30082 .coefficient)
      LeftBound29895.bound (LeftBound29895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29902RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29895.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29895.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29912.bound, LeftBound29895.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29912.bound, LeftBound29895.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29912.actual selector witness, LeftBound29895.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30083

namespace LeftBound30086
def owner : Owner := ⟨.program ⟨214⟩, ⟨26397⟩⟩
def transferEvent : Nat := 30086
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 30080 .summary, .result 29902 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 30080 .summary)
      LeftBound29914.bound (LeftBound29914.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20407⟩⟩) (rawTerms := some (Proof.Events117.exact30080RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29902 .summary)
      LeftBound29897.bound (LeftBound29897.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26396⟩⟩) (rawTerms := some (Proof.Events116.exact29902RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29897.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29914.bound, LeftBound29897.bound]
def bound : CoeffClass := .finite ⟨1291889174379421642752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29914.bound, LeftBound29897.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29914.actual selector witness, LeftBound29897.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30086

namespace LeftBound30090
def owner : Owner := ⟨.program ⟨214⟩, ⟨26607⟩⟩
def transferEvent : Nat := 30090
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 30088 .coefficient, .predecessor 1 30089 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30088 .coefficient)
      LeftBound30083.bound (LeftBound30083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30087RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30083.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30083.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30089 .coefficient)
      LeftBound29601.bound (LeftBound29601.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29605RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29601.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29601.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30083.bound, LeftBound29601.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30083.bound, LeftBound29601.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30083.actual selector witness, LeftBound29601.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30090

namespace LeftBound30091
def owner : Owner := ⟨.program ⟨214⟩, ⟨26607⟩⟩
def transferEvent : Nat := 30091
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 30087 .summary, .result 29605 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 30087 .summary)
      LeftBound30086.bound (LeftBound30086.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26397⟩⟩) (rawTerms := some (Proof.Events117.exact30087RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30086.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29605 .summary)
      LeftBound29604.bound (LeftBound29604.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26606⟩⟩) (rawTerms := some (Proof.Events115.exact29605RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29604.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30086.bound, LeftBound29604.bound]
def bound : CoeffClass := .finite ⟨2583789554981353578496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30086.bound, LeftBound29604.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30086.actual selector witness, LeftBound29604.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30091

namespace LeftBound30095
def owner : Owner := ⟨.program ⟨214⟩, ⟨26824⟩⟩
def transferEvent : Nat := 30095
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 30093 .coefficient, .predecessor 1 30094 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30093 .coefficient)
      LeftBound30090.bound (LeftBound30090.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30092RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30090.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30090.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30094 .coefficient)
      LeftBound29119.bound (LeftBound29119.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29123RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29119.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29119.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30090.bound, LeftBound29119.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30090.bound, LeftBound29119.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30090.actual selector witness, LeftBound29119.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30095

namespace LeftBound30096
def owner : Owner := ⟨.program ⟨214⟩, ⟨26824⟩⟩
def transferEvent : Nat := 30096
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 30092 .summary, .result 29123 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 30092 .summary)
      LeftBound30091.bound (LeftBound30091.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26607⟩⟩) (rawTerms := some (Proof.Events117.exact30092RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30091.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29123 .summary)
      LeftBound29122.bound (LeftBound29122.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26823⟩⟩) (rawTerms := some (Proof.Events113.exact29123RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29122.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30091.bound, LeftBound29122.bound]
def bound : CoeffClass := .finite ⟨3875701141805795807232, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30091.bound, LeftBound29122.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30091.actual selector witness, LeftBound29122.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30096

namespace LeftBound30100
def owner : Owner := ⟨.program ⟨214⟩, ⟨27041⟩⟩
def transferEvent : Nat := 30100
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 30098 .coefficient, .predecessor 1 30099 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30098 .coefficient)
      LeftBound30095.bound (LeftBound30095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30097RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30095.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30095.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30099 .coefficient)
      LeftBound28637.bound (LeftBound28637.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28641RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28637.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28637.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30095.bound, LeftBound28637.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30095.bound, LeftBound28637.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30095.actual selector witness, LeftBound28637.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30100

namespace LeftBound30101
def owner : Owner := ⟨.program ⟨214⟩, ⟨27041⟩⟩
def transferEvent : Nat := 30101
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 30097 .summary, .result 28641 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 30097 .summary)
      LeftBound30096.bound (LeftBound30096.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26824⟩⟩) (rawTerms := some (Proof.Events117.exact30097RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30096.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28641 .summary)
      LeftBound28640.bound (LeftBound28640.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27040⟩⟩) (rawTerms := some (Proof.Events111.exact28641RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28640.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30096.bound, LeftBound28640.bound]
def bound : CoeffClass := .finite ⟨5167635141075258621952, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30096.bound, LeftBound28640.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30096.actual selector witness, LeftBound28640.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30101

namespace LeftBound30105
def owner : Owner := ⟨.program ⟨214⟩, ⟨27258⟩⟩
def transferEvent : Nat := 30105
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 30103 .coefficient, .predecessor 1 30104 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30103 .coefficient)
      LeftBound30100.bound (LeftBound30100.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30102RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30100.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30100.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30104 .coefficient)
      LeftBound28155.bound (LeftBound28155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact28159RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28155.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28155.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30100.bound, LeftBound28155.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30100.bound, LeftBound28155.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30100.actual selector witness, LeftBound28155.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30105

namespace LeftBound30106
def owner : Owner := ⟨.program ⟨214⟩, ⟨27258⟩⟩
def transferEvent : Nat := 30106
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 30102 .summary, .result 28159 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 30102 .summary)
      LeftBound30101.bound (LeftBound30101.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27041⟩⟩) (rawTerms := some (Proof.Events117.exact30102RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30101.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28159 .summary)
      LeftBound28158.bound (LeftBound28158.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27257⟩⟩) (rawTerms := some (Proof.Events109.exact28159RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28158.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30101.bound, LeftBound28158.bound]
def bound : CoeffClass := .finite ⟨6459613965234762608640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30101.bound, LeftBound28158.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30101.actual selector witness, LeftBound28158.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30106

namespace LeftBound30110
def owner : Owner := ⟨.program ⟨214⟩, ⟨27475⟩⟩
def transferEvent : Nat := 30110
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 30108 .coefficient, .predecessor 1 30109 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30108 .coefficient)
      LeftBound30105.bound (LeftBound30105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30107RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30105.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30105.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30109 .coefficient)
      LeftBound27673.bound (LeftBound27673.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27673.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27673.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30105.bound, LeftBound27673.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30105.bound, LeftBound27673.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30105.actual selector witness, LeftBound27673.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30110

namespace LeftBound30111
def owner : Owner := ⟨.program ⟨214⟩, ⟨27475⟩⟩
def transferEvent : Nat := 30111
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 30107 .summary, .result 27677 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 30107 .summary)
      LeftBound30106.bound (LeftBound30106.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27258⟩⟩) (rawTerms := some (Proof.Events117.exact30107RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30106.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27677 .summary)
      LeftBound27676.bound (LeftBound27676.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27474⟩⟩) (rawTerms := some (Proof.Events108.exact27677RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27676.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30106.bound, LeftBound27676.bound]
def bound : CoeffClass := .finite ⟨7751615201839287181312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30106.bound, LeftBound27676.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30106.actual selector witness, LeftBound27676.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30111

namespace LeftBound30115
def owner : Owner := ⟨.program ⟨214⟩, ⟨27692⟩⟩
def transferEvent : Nat := 30115
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 30113 .coefficient, .predecessor 1 30114 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30113 .coefficient)
      LeftBound30110.bound (LeftBound30110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30112RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30110.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30110.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30114 .coefficient)
      LeftBound27191.bound (LeftBound27191.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27191.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27191.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30110.bound, LeftBound27191.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30110.bound, LeftBound27191.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30110.actual selector witness, LeftBound27191.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30115

namespace LeftBound30116
def owner : Owner := ⟨.program ⟨214⟩, ⟨27692⟩⟩
def transferEvent : Nat := 30116
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 30112 .summary, .result 27195 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 30112 .summary)
      LeftBound30111.bound (LeftBound30111.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27475⟩⟩) (rawTerms := some (Proof.Events117.exact30112RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound30111.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27195 .summary)
      LeftBound27194.bound (LeftBound27194.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27691⟩⟩) (rawTerms := some (Proof.Events106.exact27195RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27194.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30111.bound, LeftBound27194.bound]
def bound : CoeffClass := .finite ⟨9043661263333852925952, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30111.bound, LeftBound27194.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30111.actual selector witness, LeftBound27194.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30116

namespace LeftBound30120
def owner : Owner := ⟨.program ⟨214⟩, ⟨27909⟩⟩
def transferEvent : Nat := 30120
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 30118 .coefficient, .predecessor 1 30119 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 30118 .coefficient)
      LeftBound30115.bound (LeftBound30115.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events117.exact30117RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30115.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30115.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 30119 .coefficient)
      LeftBound26709.bound (LeftBound26709.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26709.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26709.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound30115.bound, LeftBound26709.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound30115.bound, LeftBound26709.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound30115.actual selector witness, LeftBound26709.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound30120

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
