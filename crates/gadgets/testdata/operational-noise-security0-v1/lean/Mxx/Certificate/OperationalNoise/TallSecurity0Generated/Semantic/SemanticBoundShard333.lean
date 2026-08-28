import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard314
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard315
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard316
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard317
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard318
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard319
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard321
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard322
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard323
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard332

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound50218
def owner : Owner := ⟨.program ⟨214⟩, ⟨27673⟩⟩
def transferEvent : Nat := 50218
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50216 .coefficient, .predecessor 1 50217 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50216 .coefficient)
      LeftBound50213.bound (LeftBound50213.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50213.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50213.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50217 .coefficient)
      LeftBound48879.bound (LeftBound48879.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48886RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48879.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48879.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50213.bound, LeftBound48879.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50213.bound, LeftBound48879.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50213.actual selector witness, LeftBound48879.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50218

namespace LeftBound50219
def owner : Owner := ⟨.program ⟨214⟩, ⟨27673⟩⟩
def transferEvent : Nat := 50219
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50215 .summary, .result 48886 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50215 .summary)
      LeftBound50214.bound (LeftBound50214.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27456⟩⟩) (rawTerms := some (Proof.Events196.exact50215RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50214.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48886 .summary)
      LeftBound48881.bound (LeftBound48881.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27672⟩⟩) (rawTerms := some (Proof.Events190.exact48886RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound48881.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50214.bound, LeftBound48881.bound]
def bound : CoeffClass := .finite ⟨33190381535015453579890655284, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50214.bound, LeftBound48881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50214.actual selector witness, LeftBound48881.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50219

namespace LeftBound50223
def owner : Owner := ⟨.program ⟨214⟩, ⟨27890⟩⟩
def transferEvent : Nat := 50223
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50221 .coefficient, .predecessor 1 50222 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50221 .coefficient)
      LeftBound50218.bound (LeftBound50218.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50220RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50218.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50218.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50222 .coefficient)
      LeftBound48667.bound (LeftBound48667.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events190.exact48674RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48667.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48667.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50218.bound, LeftBound48667.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50218.bound, LeftBound48667.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50218.actual selector witness, LeftBound48667.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50223

namespace LeftBound50224
def owner : Owner := ⟨.program ⟨214⟩, ⟨27890⟩⟩
def transferEvent : Nat := 50224
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50220 .summary, .result 48674 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50220 .summary)
      LeftBound50219.bound (LeftBound50219.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27673⟩⟩) (rawTerms := some (Proof.Events196.exact50220RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50219.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48674 .summary)
      LeftBound48669.bound (LeftBound48669.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27889⟩⟩) (rawTerms := some (Proof.Events190.exact48674RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound48669.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50219.bound, LeftBound48669.bound]
def bound : CoeffClass := .finite ⟨37932293507469318446662025268, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50219.bound, LeftBound48669.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50219.actual selector witness, LeftBound48669.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50224

namespace LeftBound50228
def owner : Owner := ⟨.program ⟨214⟩, ⟨28107⟩⟩
def transferEvent : Nat := 50228
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50226 .coefficient, .predecessor 1 50227 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50226 .coefficient)
      LeftBound50223.bound (LeftBound50223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50223.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50223.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50227 .coefficient)
      LeftBound48455.bound (LeftBound48455.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48455.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48455.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50223.bound, LeftBound48455.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50223.bound, LeftBound48455.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50223.actual selector witness, LeftBound48455.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50228

namespace LeftBound50229
def owner : Owner := ⟨.program ⟨214⟩, ⟨28107⟩⟩
def transferEvent : Nat := 50229
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50225 .summary, .result 48462 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50225 .summary)
      LeftBound50224.bound (LeftBound50224.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27890⟩⟩) (rawTerms := some (Proof.Events196.exact50225RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50224.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48462 .summary)
      LeftBound48457.bound (LeftBound48457.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28106⟩⟩) (rawTerms := some (Proof.Events189.exact48462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound48457.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50224.bound, LeftBound48457.bound]
def bound : CoeffClass := .finite ⟨42674369987986832655214706740, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50224.bound, LeftBound48457.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50224.actual selector witness, LeftBound48457.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50229

namespace LeftBound50233
def owner : Owner := ⟨.program ⟨214⟩, ⟨28324⟩⟩
def transferEvent : Nat := 50233
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50231 .coefficient, .predecessor 1 50232 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50231 .coefficient)
      LeftBound50228.bound (LeftBound50228.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50230RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50228.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50228.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50232 .coefficient)
      LeftBound48243.bound (LeftBound48243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events188.exact48250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48243.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48243.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50228.bound, LeftBound48243.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50228.bound, LeftBound48243.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50228.actual selector witness, LeftBound48243.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50233

namespace LeftBound50234
def owner : Owner := ⟨.program ⟨214⟩, ⟨28324⟩⟩
def transferEvent : Nat := 50234
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50230 .summary, .result 48250 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50230 .summary)
      LeftBound50229.bound (LeftBound50229.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28107⟩⟩) (rawTerms := some (Proof.Events196.exact50230RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50229.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48250 .summary)
      LeftBound48245.bound (LeftBound48245.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28323⟩⟩) (rawTerms := some (Proof.Events188.exact48250RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound48245.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50229.bound, LeftBound48245.bound]
def bound : CoeffClass := .finite ⟨47416693230599820876439355444, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50229.bound, LeftBound48245.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50229.actual selector witness, LeftBound48245.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50234

namespace LeftBound50238
def owner : Owner := ⟨.program ⟨214⟩, ⟨28541⟩⟩
def transferEvent : Nat := 50238
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50236 .coefficient, .predecessor 1 50237 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50236 .coefficient)
      LeftBound50233.bound (LeftBound50233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50235RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50233.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50233.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50237 .coefficient)
      LeftBound48031.bound (LeftBound48031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events187.exact48038RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48031.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48031.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50233.bound, LeftBound48031.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50233.bound, LeftBound48031.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50233.actual selector witness, LeftBound48031.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50238

namespace LeftBound50239
def owner : Owner := ⟨.program ⟨214⟩, ⟨28541⟩⟩
def transferEvent : Nat := 50239
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50235 .summary, .result 48038 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50235 .summary)
      LeftBound50234.bound (LeftBound50234.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28324⟩⟩) (rawTerms := some (Proof.Events196.exact50235RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50234.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48038 .summary)
      LeftBound48033.bound (LeftBound48033.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28540⟩⟩) (rawTerms := some (Proof.Events187.exact48038RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound48033.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50234.bound, LeftBound48033.bound]
def bound : CoeffClass := .finite ⟨52159098727244633768554659892, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50234.bound, LeftBound48033.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50234.actual selector witness, LeftBound48033.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50239

namespace LeftBound50243
def owner : Owner := ⟨.program ⟨214⟩, ⟨28758⟩⟩
def transferEvent : Nat := 50243
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50241 .coefficient, .predecessor 1 50242 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50241 .coefficient)
      LeftBound50238.bound (LeftBound50238.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50238.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50238.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50242 .coefficient)
      LeftBound47819.bound (LeftBound47819.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events186.exact47826RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47819.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47819.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50238.bound, LeftBound47819.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50238.bound, LeftBound47819.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50238.actual selector witness, LeftBound47819.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50243

namespace LeftBound50244
def owner : Owner := ⟨.program ⟨214⟩, ⟨28758⟩⟩
def transferEvent : Nat := 50244
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50240 .summary, .result 47826 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50240 .summary)
      LeftBound50239.bound (LeftBound50239.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28541⟩⟩) (rawTerms := some (Proof.Events196.exact50240RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50239.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 47826 .summary)
      LeftBound47821.bound (LeftBound47821.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28757⟩⟩) (rawTerms := some (Proof.Events186.exact47826RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound47821.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50239.bound, LeftBound47821.bound]
def bound : CoeffClass := .finite ⟨56901750985984920673341931572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50239.bound, LeftBound47821.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50239.actual selector witness, LeftBound47821.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50244

namespace LeftBound50248
def owner : Owner := ⟨.program ⟨214⟩, ⟨28975⟩⟩
def transferEvent : Nat := 50248
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50246 .coefficient, .predecessor 1 50247 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50246 .coefficient)
      LeftBound50243.bound (LeftBound50243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50243.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50247 .coefficient)
      LeftBound47607.bound (LeftBound47607.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47614RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47607.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47607.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50243.bound, LeftBound47607.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50243.bound, LeftBound47607.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50243.actual selector witness, LeftBound47607.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50248

namespace LeftBound50249
def owner : Owner := ⟨.program ⟨214⟩, ⟨28975⟩⟩
def transferEvent : Nat := 50249
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50245 .summary, .result 47614 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50245 .summary)
      LeftBound50244.bound (LeftBound50244.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28758⟩⟩) (rawTerms := some (Proof.Events196.exact50245RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50244.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 47614 .summary)
      LeftBound47609.bound (LeftBound47609.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28974⟩⟩) (rawTerms := some (Proof.Events185.exact47614RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound47609.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50244.bound, LeftBound47609.bound]
def bound : CoeffClass := .finite ⟨61644567752788856919910514740, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50244.bound, LeftBound47609.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50244.actual selector witness, LeftBound47609.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50249

namespace LeftBound50253
def owner : Owner := ⟨.program ⟨214⟩, ⟨29192⟩⟩
def transferEvent : Nat := 50253
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50251 .coefficient, .predecessor 1 50252 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50251 .coefficient)
      LeftBound50248.bound (LeftBound50248.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50248.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50248.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50252 .coefficient)
      LeftBound47395.bound (LeftBound47395.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47395.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47395.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50248.bound, LeftBound47395.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50248.bound, LeftBound47395.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50248.actual selector witness, LeftBound47395.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50253

namespace LeftBound50254
def owner : Owner := ⟨.program ⟨214⟩, ⟨29192⟩⟩
def transferEvent : Nat := 50254
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50250 .summary, .result 47402 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50250 .summary)
      LeftBound50249.bound (LeftBound50249.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28975⟩⟩) (rawTerms := some (Proof.Events196.exact50250RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50249.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 47402 .summary)
      LeftBound47397.bound (LeftBound47397.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29191⟩⟩) (rawTerms := some (Proof.Events185.exact47402RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound47397.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50249.bound, LeftBound47397.bound]
def bound : CoeffClass := .finite ⟨66387466773624617837369753652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50249.bound, LeftBound47397.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50249.actual selector witness, LeftBound47397.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50254

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
