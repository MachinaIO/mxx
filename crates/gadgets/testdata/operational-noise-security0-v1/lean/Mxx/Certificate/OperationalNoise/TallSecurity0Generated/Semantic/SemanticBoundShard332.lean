import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard128
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard325
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard326
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard327
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard329
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard330
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard331

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound50175
def owner : Owner := ⟨.program ⟨214⟩, ⟨7762⟩⟩
def transferEvent : Nat := 50175
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50173 .coefficient, .predecessor 1 50174 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50173 .coefficient)
      LeftBound50171.bound (LeftBound50171.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events195.exact50172RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50171.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50171.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50174 .coefficient)
      LeftBound20907.bound (LeftBound20907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20907.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50171.bound, LeftBound20907.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50171.bound, LeftBound20907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50171.actual selector witness, LeftBound20907.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50175

namespace LeftBound50176
def owner : Owner := ⟨.program ⟨214⟩, ⟨7762⟩⟩
def transferEvent : Nat := 50176
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨74⟩⟩]⟩ [⟨.result 20908 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20908 .coefficient)
      LeftBound20907.bound (LeftBound20907.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨74⟩⟩) (rawTerms := some (Proof.Events081.exact20908RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20907.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound20907.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound20907.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound50176

namespace LeftBound50181
def owner : Owner := ⟨.program ⟨214⟩, ⟨7810⟩⟩
def transferEvent : Nat := 50181
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50179 .coefficient, .predecessor 1 50180 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50179 .coefficient)
      LeftBound50175.bound (LeftBound50175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50175.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50175.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50180 .coefficient)
      LeftBound50175.bound (LeftBound50175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50175.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50175.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50175.bound, LeftBound50175.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50175.bound, LeftBound50175.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50175.actual selector witness, LeftBound50175.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50181

namespace LeftBound50184
def owner : Owner := ⟨.program ⟨214⟩, ⟨7810⟩⟩
def transferEvent : Nat := 50184
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50178 .summary, .result 50178 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50178 .summary)
      LeftBound50176.bound (LeftBound50176.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7762⟩⟩) (rawTerms := some (Proof.Events196.exact50178RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50178 .summary)
      LeftBound50176.bound (LeftBound50176.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7762⟩⟩) (rawTerms := some (Proof.Events196.exact50178RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50176.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50176.bound, LeftBound50176.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50176.bound, LeftBound50176.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50176.actual selector witness, LeftBound50176.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50184

namespace LeftBound50188
def owner : Owner := ⟨.program ⟨214⟩, ⟨26380⟩⟩
def transferEvent : Nat := 50188
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50186 .coefficient, .predecessor 1 50187 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50186 .coefficient)
      LeftBound50181.bound (LeftBound50181.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50185RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50181.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50181.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50187 .coefficient)
      LeftBound50151.bound (LeftBound50151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events195.exact50158RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50151.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50181.bound, LeftBound50151.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50181.bound, LeftBound50151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50181.actual selector witness, LeftBound50151.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50188

namespace LeftBound50189
def owner : Owner := ⟨.program ⟨214⟩, ⟨26380⟩⟩
def transferEvent : Nat := 50189
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50185 .summary, .result 50158 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50185 .summary)
      LeftBound50184.bound (LeftBound50184.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7810⟩⟩) (rawTerms := some (Proof.Events196.exact50185RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50184.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50158 .summary)
      LeftBound50153.bound (LeftBound50153.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26379⟩⟩) (rawTerms := some (Proof.Events195.exact50158RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50153.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50184.bound, LeftBound50153.bound]
def bound : CoeffClass := .finite ⟨4741253940199267499646124084, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50184.bound, LeftBound50153.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50184.actual selector witness, LeftBound50153.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50189

namespace LeftBound50193
def owner : Owner := ⟨.program ⟨214⟩, ⟨26588⟩⟩
def transferEvent : Nat := 50193
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50191 .coefficient, .predecessor 1 50192 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50191 .coefficient)
      LeftBound50188.bound (LeftBound50188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50190RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50188.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50192 .coefficient)
      LeftBound49939.bound (LeftBound49939.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events195.exact49946RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49939.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49939.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50188.bound, LeftBound49939.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50188.bound, LeftBound49939.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50188.actual selector witness, LeftBound49939.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50193

namespace LeftBound50194
def owner : Owner := ⟨.program ⟨214⟩, ⟨26588⟩⟩
def transferEvent : Nat := 50194
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50190 .summary, .result 49946 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50190 .summary)
      LeftBound50189.bound (LeftBound50189.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26380⟩⟩) (rawTerms := some (Proof.Events196.exact50190RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49946 .summary)
      LeftBound49941.bound (LeftBound49941.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26587⟩⟩) (rawTerms := some (Proof.Events195.exact49946RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound49941.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50189.bound, LeftBound49941.bound]
def bound : CoeffClass := .finite ⟨9482549007414447334737575988, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50189.bound, LeftBound49941.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50189.actual selector witness, LeftBound49941.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50194

namespace LeftBound50198
def owner : Owner := ⟨.program ⟨214⟩, ⟨26805⟩⟩
def transferEvent : Nat := 50198
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50196 .coefficient, .predecessor 1 50197 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50196 .coefficient)
      LeftBound50193.bound (LeftBound50193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50193.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50197 .coefficient)
      LeftBound49727.bound (LeftBound49727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49734RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49727.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49727.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50193.bound, LeftBound49727.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50193.bound, LeftBound49727.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50193.actual selector witness, LeftBound49727.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50198

namespace LeftBound50199
def owner : Owner := ⟨.program ⟨214⟩, ⟨26805⟩⟩
def transferEvent : Nat := 50199
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50195 .summary, .result 49734 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50195 .summary)
      LeftBound50194.bound (LeftBound50194.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26588⟩⟩) (rawTerms := some (Proof.Events196.exact50195RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50194.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49734 .summary)
      LeftBound49729.bound (LeftBound49729.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26804⟩⟩) (rawTerms := some (Proof.Events194.exact49734RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound49729.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50194.bound, LeftBound49729.bound]
def bound : CoeffClass := .finite ⟨14223885201645539505274355764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50194.bound, LeftBound49729.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50194.actual selector witness, LeftBound49729.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50199

namespace LeftBound50203
def owner : Owner := ⟨.program ⟨214⟩, ⟨27022⟩⟩
def transferEvent : Nat := 50203
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50201 .coefficient, .predecessor 1 50202 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50201 .coefficient)
      LeftBound50198.bound (LeftBound50198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50200RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50198.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50198.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50202 .coefficient)
      LeftBound49515.bound (LeftBound49515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events193.exact49522RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49515.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50198.bound, LeftBound49515.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50198.bound, LeftBound49515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50198.actual selector witness, LeftBound49515.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50203

namespace LeftBound50204
def owner : Owner := ⟨.program ⟨214⟩, ⟨27022⟩⟩
def transferEvent : Nat := 50204
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50200 .summary, .result 49522 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50200 .summary)
      LeftBound50199.bound (LeftBound50199.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26805⟩⟩) (rawTerms := some (Proof.Events196.exact50200RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50199.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49522 .summary)
      LeftBound49517.bound (LeftBound49517.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27021⟩⟩) (rawTerms := some (Proof.Events193.exact49522RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound49517.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50199.bound, LeftBound49517.bound]
def bound : CoeffClass := .finite ⟨18965303649908456346701791284, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50199.bound, LeftBound49517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50199.actual selector witness, LeftBound49517.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50204

namespace LeftBound50208
def owner : Owner := ⟨.program ⟨214⟩, ⟨27239⟩⟩
def transferEvent : Nat := 50208
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50206 .coefficient, .predecessor 1 50207 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50206 .coefficient)
      LeftBound50203.bound (LeftBound50203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50203.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50203.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50207 .coefficient)
      LeftBound49303.bound (LeftBound49303.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events192.exact49310RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49303.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49303.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50203.bound, LeftBound49303.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50203.bound, LeftBound49303.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50203.actual selector witness, LeftBound49303.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50208

namespace LeftBound50209
def owner : Owner := ⟨.program ⟨214⟩, ⟨27239⟩⟩
def transferEvent : Nat := 50209
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50205 .summary, .result 49310 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50205 .summary)
      LeftBound50204.bound (LeftBound50204.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27022⟩⟩) (rawTerms := some (Proof.Events196.exact50205RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50204.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49310 .summary)
      LeftBound49305.bound (LeftBound49305.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27238⟩⟩) (rawTerms := some (Proof.Events192.exact49310RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound49305.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50204.bound, LeftBound49305.bound]
def bound : CoeffClass := .finite ⟨23706886606235022529910538292, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50204.bound, LeftBound49305.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50204.actual selector witness, LeftBound49305.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50209

namespace LeftBound50213
def owner : Owner := ⟨.program ⟨214⟩, ⟨27456⟩⟩
def transferEvent : Nat := 50213
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50211 .coefficient, .predecessor 1 50212 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50211 .coefficient)
      LeftBound50208.bound (LeftBound50208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events196.exact50210RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50208.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50208.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50212 .coefficient)
      LeftBound49091.bound (LeftBound49091.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events191.exact49098RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49091.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49091.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50208.bound, LeftBound49091.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50208.bound, LeftBound49091.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50208.actual selector witness, LeftBound49091.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50213

namespace LeftBound50214
def owner : Owner := ⟨.program ⟨214⟩, ⟨27456⟩⟩
def transferEvent : Nat := 50214
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50210 .summary, .result 49098 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50210 .summary)
      LeftBound50209.bound (LeftBound50209.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27239⟩⟩) (rawTerms := some (Proof.Events196.exact50210RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50209.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49098 .summary)
      LeftBound49093.bound (LeftBound49093.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27455⟩⟩) (rawTerms := some (Proof.Events191.exact49098RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound49093.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50209.bound, LeftBound49093.bound]
def bound : CoeffClass := .finite ⟨28448551816593413384009941044, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50209.bound, LeftBound49093.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50209.actual selector witness, LeftBound49093.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50214

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
