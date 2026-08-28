import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard461
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard465
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard469
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard472
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard476
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard480
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard483
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard487
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard505

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound73985
def owner : Owner := ⟨.program ⟨214⟩, ⟨27423⟩⟩
def transferEvent : Nat := 73985
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 73983 .coefficient, .predecessor 1 73984 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73983 .coefficient)
      LeftBound73980.bound (LeftBound73980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73982RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73980.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73984 .coefficient)
      LeftBound71548.bound (LeftBound71548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events279.exact71552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71548.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71548.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73980.bound, LeftBound71548.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73980.bound, LeftBound71548.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73980.actual selector witness, LeftBound71548.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73985

namespace LeftBound73986
def owner : Owner := ⟨.program ⟨214⟩, ⟨27423⟩⟩
def transferEvent : Nat := 73986
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 73982 .summary, .result 71552 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73982 .summary)
      LeftBound73981.bound (LeftBound73981.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27206⟩⟩) (rawTerms := some (Proof.Events288.exact73982RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73981.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71552 .summary)
      LeftBound71551.bound (LeftBound71551.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27422⟩⟩) (rawTerms := some (Proof.Events279.exact71552RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound71551.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73981.bound, LeftBound71551.bound]
def bound : CoeffClass := .finite ⟨7751615201839287181312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73981.bound, LeftBound71551.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73981.actual selector witness, LeftBound71551.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73986

namespace LeftBound73990
def owner : Owner := ⟨.program ⟨214⟩, ⟨27640⟩⟩
def transferEvent : Nat := 73990
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 73988 .coefficient, .predecessor 1 73989 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73988 .coefficient)
      LeftBound73985.bound (LeftBound73985.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events289.exact73987RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73985.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73985.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73989 .coefficient)
      LeftBound71066.bound (LeftBound71066.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71070RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71066.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71066.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73985.bound, LeftBound71066.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73985.bound, LeftBound71066.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73985.actual selector witness, LeftBound71066.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73990

namespace LeftBound73991
def owner : Owner := ⟨.program ⟨214⟩, ⟨27640⟩⟩
def transferEvent : Nat := 73991
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 73987 .summary, .result 71070 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73987 .summary)
      LeftBound73986.bound (LeftBound73986.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27423⟩⟩) (rawTerms := some (Proof.Events289.exact73987RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73986.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71070 .summary)
      LeftBound71069.bound (LeftBound71069.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27639⟩⟩) (rawTerms := some (Proof.Events277.exact71070RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound71069.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73986.bound, LeftBound71069.bound]
def bound : CoeffClass := .finite ⟨9043661263333852925952, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73986.bound, LeftBound71069.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73986.actual selector witness, LeftBound71069.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73991

namespace LeftBound73995
def owner : Owner := ⟨.program ⟨214⟩, ⟨27857⟩⟩
def transferEvent : Nat := 73995
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 73993 .coefficient, .predecessor 1 73994 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73993 .coefficient)
      LeftBound73990.bound (LeftBound73990.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events289.exact73992RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73990.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73990.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73994 .coefficient)
      LeftBound70584.bound (LeftBound70584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70584.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73990.bound, LeftBound70584.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73990.bound, LeftBound70584.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73990.actual selector witness, LeftBound70584.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73995

namespace LeftBound73996
def owner : Owner := ⟨.program ⟨214⟩, ⟨27857⟩⟩
def transferEvent : Nat := 73996
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 73992 .summary, .result 70588 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73992 .summary)
      LeftBound73991.bound (LeftBound73991.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27640⟩⟩) (rawTerms := some (Proof.Events289.exact73992RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73991.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70588 .summary)
      LeftBound70587.bound (LeftBound70587.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27856⟩⟩) (rawTerms := some (Proof.Events275.exact70588RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70587.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73991.bound, LeftBound70587.bound]
def bound : CoeffClass := .finite ⟨10335729737273439256576, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73991.bound, LeftBound70587.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73991.actual selector witness, LeftBound70587.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73996

namespace LeftBound74000
def owner : Owner := ⟨.program ⟨214⟩, ⟨28074⟩⟩
def transferEvent : Nat := 74000
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 73998 .coefficient, .predecessor 1 73999 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73998 .coefficient)
      LeftBound73995.bound (LeftBound73995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events289.exact73997RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73995.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73995.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73999 .coefficient)
      LeftBound70102.bound (LeftBound70102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70102.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70102.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73995.bound, LeftBound70102.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73995.bound, LeftBound70102.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73995.actual selector witness, LeftBound70102.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74000

namespace LeftBound74001
def owner : Owner := ⟨.program ⟨214⟩, ⟨28074⟩⟩
def transferEvent : Nat := 74001
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 73997 .summary, .result 70106 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73997 .summary)
      LeftBound73996.bound (LeftBound73996.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27857⟩⟩) (rawTerms := some (Proof.Events289.exact73997RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73996.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70106 .summary)
      LeftBound70105.bound (LeftBound70105.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28073⟩⟩) (rawTerms := some (Proof.Events273.exact70106RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70105.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73996.bound, LeftBound70105.bound]
def bound : CoeffClass := .finite ⟨11627843036103066759168, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73996.bound, LeftBound70105.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73996.actual selector witness, LeftBound70105.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74001

namespace LeftBound74005
def owner : Owner := ⟨.program ⟨214⟩, ⟨28291⟩⟩
def transferEvent : Nat := 74005
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 74003 .coefficient, .predecessor 1 74004 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 74003 .coefficient)
      LeftBound74000.bound (LeftBound74000.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events289.exact74002RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74000.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74000.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 74004 .coefficient)
      LeftBound69620.bound (LeftBound69620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69624RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69620.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69620.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74000.bound, LeftBound69620.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74000.bound, LeftBound69620.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound74000.actual selector witness, LeftBound69620.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74005

namespace LeftBound74006
def owner : Owner := ⟨.program ⟨214⟩, ⟨28291⟩⟩
def transferEvent : Nat := 74006
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 74002 .summary, .result 69624 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 74002 .summary)
      LeftBound74001.bound (LeftBound74001.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28074⟩⟩) (rawTerms := some (Proof.Events289.exact74002RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound74001.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 69624 .summary)
      LeftBound69623.bound (LeftBound69623.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28290⟩⟩) (rawTerms := some (Proof.Events271.exact69624RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69623.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74001.bound, LeftBound69623.bound]
def bound : CoeffClass := .finite ⟨12920023572267756019712, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74001.bound, LeftBound69623.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound74001.actual selector witness, LeftBound69623.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74006

namespace LeftBound74010
def owner : Owner := ⟨.program ⟨214⟩, ⟨28508⟩⟩
def transferEvent : Nat := 74010
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 74008 .coefficient, .predecessor 1 74009 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 74008 .coefficient)
      LeftBound74005.bound (LeftBound74005.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events289.exact74007RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74005.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 74009 .coefficient)
      LeftBound69138.bound (LeftBound69138.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events270.exact69142RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69138.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69138.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74005.bound, LeftBound69138.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74005.bound, LeftBound69138.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound74005.actual selector witness, LeftBound69138.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74010

namespace LeftBound74011
def owner : Owner := ⟨.program ⟨214⟩, ⟨28508⟩⟩
def transferEvent : Nat := 74011
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 74007 .summary, .result 69142 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 74007 .summary)
      LeftBound74006.bound (LeftBound74006.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28291⟩⟩) (rawTerms := some (Proof.Events289.exact74007RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound74006.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 69142 .summary)
      LeftBound69141.bound (LeftBound69141.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28507⟩⟩) (rawTerms := some (Proof.Events270.exact69142RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69141.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74006.bound, LeftBound69141.bound]
def bound : CoeffClass := .finite ⟨14212226520877465866240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74006.bound, LeftBound69141.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound74006.actual selector witness, LeftBound69141.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74011

namespace LeftBound74015
def owner : Owner := ⟨.program ⟨214⟩, ⟨28725⟩⟩
def transferEvent : Nat := 74015
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 74013 .coefficient, .predecessor 1 74014 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 74013 .coefficient)
      LeftBound74010.bound (LeftBound74010.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events289.exact74012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74010.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 74014 .coefficient)
      LeftBound68656.bound (LeftBound68656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68660RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68656.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68656.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74010.bound, LeftBound68656.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74010.bound, LeftBound68656.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound74010.actual selector witness, LeftBound68656.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74015

namespace LeftBound74016
def owner : Owner := ⟨.program ⟨214⟩, ⟨28725⟩⟩
def transferEvent : Nat := 74016
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 74012 .summary, .result 68660 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 74012 .summary)
      LeftBound74011.bound (LeftBound74011.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28508⟩⟩) (rawTerms := some (Proof.Events289.exact74012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound74011.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68660 .summary)
      LeftBound68659.bound (LeftBound68659.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28724⟩⟩) (rawTerms := some (Proof.Events268.exact68660RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68659.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74011.bound, LeftBound68659.bound]
def bound : CoeffClass := .finite ⟨15504496706822237470720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74011.bound, LeftBound68659.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound74011.actual selector witness, LeftBound68659.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74016

namespace LeftBound74020
def owner : Owner := ⟨.program ⟨214⟩, ⟨28942⟩⟩
def transferEvent : Nat := 74020
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 74018 .coefficient, .predecessor 1 74019 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 74018 .coefficient)
      LeftBound74015.bound (LeftBound74015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events289.exact74017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound74015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound74015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 74019 .coefficient)
      LeftBound68174.bound (LeftBound68174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68174.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68174.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74015.bound, LeftBound68174.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74015.bound, LeftBound68174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound74015.actual selector witness, LeftBound68174.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74020

namespace LeftBound74021
def owner : Owner := ⟨.program ⟨214⟩, ⟨28942⟩⟩
def transferEvent : Nat := 74021
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 74017 .summary, .result 68178 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 74017 .summary)
      LeftBound74016.bound (LeftBound74016.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28725⟩⟩) (rawTerms := some (Proof.Events289.exact74017RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound74016.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68178 .summary)
      LeftBound68177.bound (LeftBound68177.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28941⟩⟩) (rawTerms := some (Proof.Events266.exact68178RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68177.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound74016.bound, LeftBound68177.bound]
def bound : CoeffClass := .finite ⟨16796811717657050247168, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound74016.bound, LeftBound68177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound74016.actual selector witness, LeftBound68177.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound74021

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
