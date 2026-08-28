import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard725
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard726
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard727
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard728
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard729
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard730
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard731
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard733
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard734

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound107189
def owner : Owner := ⟨.program ⟨214⟩, ⟨7805⟩⟩
def transferEvent : Nat := 107189
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107183 .summary, .result 107183 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107183 .summary)
      LeftBound107181.bound (LeftBound107181.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7732⟩⟩) (rawTerms := some (Proof.Events418.exact107183RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107181.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107183 .summary)
      LeftBound107181.bound (LeftBound107181.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7732⟩⟩) (rawTerms := some (Proof.Events418.exact107183RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107181.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107181.bound, LeftBound107181.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107181.bound, LeftBound107181.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107181.actual selector witness, LeftBound107181.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107189

namespace LeftBound107193
def owner : Owner := ⟨.program ⟨214⟩, ⟨26324⟩⟩
def transferEvent : Nat := 107193
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107191 .coefficient, .predecessor 1 107192 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107191 .coefficient)
      LeftBound107186.bound (LeftBound107186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107190RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107186.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107186.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107192 .coefficient)
      LeftBound107156.bound (LeftBound107156.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107163RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107156.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107156.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107186.bound, LeftBound107156.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107186.bound, LeftBound107156.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107186.actual selector witness, LeftBound107156.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107193

namespace LeftBound107194
def owner : Owner := ⟨.program ⟨214⟩, ⟨26324⟩⟩
def transferEvent : Nat := 107194
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107190 .summary, .result 107163 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107190 .summary)
      LeftBound107189.bound (LeftBound107189.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7805⟩⟩) (rawTerms := some (Proof.Events418.exact107190RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107163 .summary)
      LeftBound107158.bound (LeftBound107158.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26323⟩⟩) (rawTerms := some (Proof.Events418.exact107163RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107158.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107189.bound, LeftBound107158.bound]
def bound : CoeffClass := .finite ⟨4741253940199267499646124084, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107189.bound, LeftBound107158.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107189.actual selector witness, LeftBound107158.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107194

namespace LeftBound107198
def owner : Owner := ⟨.program ⟨214⟩, ⟨26527⟩⟩
def transferEvent : Nat := 107198
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107196 .coefficient, .predecessor 1 107197 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107196 .coefficient)
      LeftBound107193.bound (LeftBound107193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107193.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107197 .coefficient)
      LeftBound106968.bound (LeftBound106968.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106968.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106968.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107193.bound, LeftBound106968.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107193.bound, LeftBound106968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107193.actual selector witness, LeftBound106968.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107198

namespace LeftBound107199
def owner : Owner := ⟨.program ⟨214⟩, ⟨26527⟩⟩
def transferEvent : Nat := 107199
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107195 .summary, .result 106975 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107195 .summary)
      LeftBound107194.bound (LeftBound107194.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26324⟩⟩) (rawTerms := some (Proof.Events418.exact107195RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107194.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106975 .summary)
      LeftBound106970.bound (LeftBound106970.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26526⟩⟩) (rawTerms := some (Proof.Events417.exact106975RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106970.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107194.bound, LeftBound106970.bound]
def bound : CoeffClass := .finite ⟨9482549007414447334737575988, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107194.bound, LeftBound106970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107194.actual selector witness, LeftBound106970.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107199

namespace LeftBound107203
def owner : Owner := ⟨.program ⟨214⟩, ⟨26744⟩⟩
def transferEvent : Nat := 107203
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107201 .coefficient, .predecessor 1 107202 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107201 .coefficient)
      LeftBound107198.bound (LeftBound107198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107200RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107198.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107198.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107202 .coefficient)
      LeftBound106780.bound (LeftBound106780.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106787RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106780.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106780.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107198.bound, LeftBound106780.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107198.bound, LeftBound106780.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107198.actual selector witness, LeftBound106780.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107203

namespace LeftBound107204
def owner : Owner := ⟨.program ⟨214⟩, ⟨26744⟩⟩
def transferEvent : Nat := 107204
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107200 .summary, .result 106787 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107200 .summary)
      LeftBound107199.bound (LeftBound107199.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26527⟩⟩) (rawTerms := some (Proof.Events418.exact107200RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107199.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106787 .summary)
      LeftBound106782.bound (LeftBound106782.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26743⟩⟩) (rawTerms := some (Proof.Events417.exact106787RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106782.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107199.bound, LeftBound106782.bound]
def bound : CoeffClass := .finite ⟨14223885201645539505274355764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107199.bound, LeftBound106782.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107199.actual selector witness, LeftBound106782.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107204

namespace LeftBound107208
def owner : Owner := ⟨.program ⟨214⟩, ⟨26961⟩⟩
def transferEvent : Nat := 107208
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107206 .coefficient, .predecessor 1 107207 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107206 .coefficient)
      LeftBound107203.bound (LeftBound107203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107203.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107203.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107207 .coefficient)
      LeftBound106592.bound (LeftBound106592.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events416.exact106599RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106592.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106592.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107203.bound, LeftBound106592.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107203.bound, LeftBound106592.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107203.actual selector witness, LeftBound106592.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107208

namespace LeftBound107209
def owner : Owner := ⟨.program ⟨214⟩, ⟨26961⟩⟩
def transferEvent : Nat := 107209
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107205 .summary, .result 106599 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107205 .summary)
      LeftBound107204.bound (LeftBound107204.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26744⟩⟩) (rawTerms := some (Proof.Events418.exact107205RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107204.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106599 .summary)
      LeftBound106594.bound (LeftBound106594.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26960⟩⟩) (rawTerms := some (Proof.Events416.exact106599RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106594.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107204.bound, LeftBound106594.bound]
def bound : CoeffClass := .finite ⟨18965303649908456346701791284, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107204.bound, LeftBound106594.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107204.actual selector witness, LeftBound106594.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107209

namespace LeftBound107213
def owner : Owner := ⟨.program ⟨214⟩, ⟨27178⟩⟩
def transferEvent : Nat := 107213
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107211 .coefficient, .predecessor 1 107212 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107211 .coefficient)
      LeftBound107208.bound (LeftBound107208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107210RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107208.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107208.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107212 .coefficient)
      LeftBound106404.bound (LeftBound106404.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events415.exact106411RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106404.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106404.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107208.bound, LeftBound106404.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107208.bound, LeftBound106404.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107208.actual selector witness, LeftBound106404.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107213

namespace LeftBound107214
def owner : Owner := ⟨.program ⟨214⟩, ⟨27178⟩⟩
def transferEvent : Nat := 107214
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107210 .summary, .result 106411 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107210 .summary)
      LeftBound107209.bound (LeftBound107209.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26961⟩⟩) (rawTerms := some (Proof.Events418.exact107210RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107209.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106411 .summary)
      LeftBound106406.bound (LeftBound106406.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27177⟩⟩) (rawTerms := some (Proof.Events415.exact106411RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106406.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107209.bound, LeftBound106406.bound]
def bound : CoeffClass := .finite ⟨23706886606235022529910538292, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107209.bound, LeftBound106406.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107209.actual selector witness, LeftBound106406.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107214

namespace LeftBound107218
def owner : Owner := ⟨.program ⟨214⟩, ⟨27395⟩⟩
def transferEvent : Nat := 107218
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107216 .coefficient, .predecessor 1 107217 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107216 .coefficient)
      LeftBound107213.bound (LeftBound107213.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107213.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107213.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107217 .coefficient)
      LeftBound106216.bound (LeftBound106216.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106216.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106216.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107213.bound, LeftBound106216.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107213.bound, LeftBound106216.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107213.actual selector witness, LeftBound106216.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107218

namespace LeftBound107219
def owner : Owner := ⟨.program ⟨214⟩, ⟨27395⟩⟩
def transferEvent : Nat := 107219
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107215 .summary, .result 106223 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107215 .summary)
      LeftBound107214.bound (LeftBound107214.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27178⟩⟩) (rawTerms := some (Proof.Events418.exact107215RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107214.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106223 .summary)
      LeftBound106218.bound (LeftBound106218.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27394⟩⟩) (rawTerms := some (Proof.Events414.exact106223RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106218.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107214.bound, LeftBound106218.bound]
def bound : CoeffClass := .finite ⟨28448551816593413384009941044, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107214.bound, LeftBound106218.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107214.actual selector witness, LeftBound106218.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107219

namespace LeftBound107223
def owner : Owner := ⟨.program ⟨214⟩, ⟨27612⟩⟩
def transferEvent : Nat := 107223
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107221 .coefficient, .predecessor 1 107222 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107221 .coefficient)
      LeftBound107218.bound (LeftBound107218.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107220RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107218.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107218.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107222 .coefficient)
      LeftBound106028.bound (LeftBound106028.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events414.exact106035RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106028.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106028.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107218.bound, LeftBound106028.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107218.bound, LeftBound106028.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107218.actual selector witness, LeftBound106028.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107223

namespace LeftBound107224
def owner : Owner := ⟨.program ⟨214⟩, ⟨27612⟩⟩
def transferEvent : Nat := 107224
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107220 .summary, .result 106035 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107220 .summary)
      LeftBound107219.bound (LeftBound107219.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27395⟩⟩) (rawTerms := some (Proof.Events418.exact107220RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107219.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106035 .summary)
      LeftBound106030.bound (LeftBound106030.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27611⟩⟩) (rawTerms := some (Proof.Events414.exact106035RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106030.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107219.bound, LeftBound106030.bound]
def bound : CoeffClass := .finite ⟨33190381535015453579890655284, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107219.bound, LeftBound106030.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107219.actual selector witness, LeftBound106030.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107224

namespace LeftBound107228
def owner : Owner := ⟨.program ⟨214⟩, ⟨27829⟩⟩
def transferEvent : Nat := 107228
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107226 .coefficient, .predecessor 1 107227 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107226 .coefficient)
      LeftBound107223.bound (LeftBound107223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107223.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107223.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107227 .coefficient)
      LeftBound105840.bound (LeftBound105840.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events413.exact105847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105840.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105840.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107223.bound, LeftBound105840.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107223.bound, LeftBound105840.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107223.actual selector witness, LeftBound105840.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107228

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
