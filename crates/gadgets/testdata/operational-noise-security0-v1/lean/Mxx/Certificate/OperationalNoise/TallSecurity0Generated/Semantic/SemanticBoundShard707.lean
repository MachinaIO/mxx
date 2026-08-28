import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard643
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard647
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard650
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard651
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard654
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard658
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard706

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound102237
def owner : Owner := ⟨.program ⟨214⟩, ⟨29137⟩⟩
def transferEvent : Nat := 102237
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 102233 .summary, .result 96531 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102233 .summary)
      LeftBound102232.bound (LeftBound102232.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28920⟩⟩) (rawTerms := some (Proof.Events399.exact102233RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102232.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96531 .summary)
      LeftBound96530.bound (LeftBound96530.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29136⟩⟩) (rawTerms := some (Proof.Events377.exact96531RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96530.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102232.bound, LeftBound96530.bound]
def bound : CoeffClass := .finite ⟨18089149140936883609600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102232.bound, LeftBound96530.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102232.actual selector witness, LeftBound96530.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102237

namespace LeftBound102241
def owner : Owner := ⟨.program ⟨214⟩, ⟨29354⟩⟩
def transferEvent : Nat := 102241
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 102239 .coefficient, .predecessor 1 102240 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102239 .coefficient)
      LeftBound102236.bound (LeftBound102236.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102238RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102236.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102236.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102240 .coefficient)
      LeftBound96093.bound (LeftBound96093.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events375.exact96097RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96093.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96093.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102236.bound, LeftBound96093.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102236.bound, LeftBound96093.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102236.actual selector witness, LeftBound96093.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102241

namespace LeftBound102242
def owner : Owner := ⟨.program ⟨214⟩, ⟨29354⟩⟩
def transferEvent : Nat := 102242
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 102238 .summary, .result 96097 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102238 .summary)
      LeftBound102237.bound (LeftBound102237.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29137⟩⟩) (rawTerms := some (Proof.Events399.exact102238RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102237.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96097 .summary)
      LeftBound96096.bound (LeftBound96096.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29353⟩⟩) (rawTerms := some (Proof.Events375.exact96097RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96096.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102237.bound, LeftBound96096.bound]
def bound : CoeffClass := .finite ⟨19381531389106758144000, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102237.bound, LeftBound96096.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102237.actual selector witness, LeftBound96096.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102242

namespace LeftBound102246
def owner : Owner := ⟨.program ⟨214⟩, ⟨29571⟩⟩
def transferEvent : Nat := 102246
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 102244 .coefficient, .predecessor 1 102245 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102244 .coefficient)
      LeftBound102241.bound (LeftBound102241.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102243RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102241.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102241.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102245 .coefficient)
      LeftBound95659.bound (LeftBound95659.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95659.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95659.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102241.bound, LeftBound95659.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102241.bound, LeftBound95659.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102241.actual selector witness, LeftBound95659.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102246

namespace LeftBound102247
def owner : Owner := ⟨.program ⟨214⟩, ⟨29571⟩⟩
def transferEvent : Nat := 102247
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 102243 .summary, .result 95663 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102243 .summary)
      LeftBound102242.bound (LeftBound102242.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29354⟩⟩) (rawTerms := some (Proof.Events399.exact102243RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95663 .summary)
      LeftBound95662.bound (LeftBound95662.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29570⟩⟩) (rawTerms := some (Proof.Events373.exact95663RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95662.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102242.bound, LeftBound95662.bound]
def bound : CoeffClass := .finite ⟨20673980874611694436352, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102242.bound, LeftBound95662.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102242.actual selector witness, LeftBound95662.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102247

namespace LeftBound102251
def owner : Owner := ⟨.program ⟨214⟩, ⟨29788⟩⟩
def transferEvent : Nat := 102251
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 102249 .coefficient, .predecessor 1 102250 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102249 .coefficient)
      LeftBound102246.bound (LeftBound102246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102250 .coefficient)
      LeftBound95225.bound (LeftBound95225.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events371.exact95229RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95225.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95225.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102246.bound, LeftBound95225.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102246.bound, LeftBound95225.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102246.actual selector witness, LeftBound95225.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102251

namespace LeftBound102252
def owner : Owner := ⟨.program ⟨214⟩, ⟨29788⟩⟩
def transferEvent : Nat := 102252
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 102248 .summary, .result 95229 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102248 .summary)
      LeftBound102247.bound (LeftBound102247.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29571⟩⟩) (rawTerms := some (Proof.Events399.exact102248RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95229 .summary)
      LeftBound95228.bound (LeftBound95228.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29787⟩⟩) (rawTerms := some (Proof.Events371.exact95229RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95228.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102247.bound, LeftBound95228.bound]
def bound : CoeffClass := .finite ⟨21966497597451692486656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102247.bound, LeftBound95228.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102247.actual selector witness, LeftBound95228.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102252

namespace LeftBound102256
def owner : Owner := ⟨.program ⟨214⟩, ⟨30065⟩⟩
def transferEvent : Nat := 102256
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 102254 .coefficient, .predecessor 1 102255 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102254 .coefficient)
      LeftBound102251.bound (LeftBound102251.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102253RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102251.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102251.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102255 .coefficient)
      LeftBound94791.bound (LeftBound94791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94795RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94791.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94791.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102251.bound, LeftBound94791.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102251.bound, LeftBound94791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102251.actual selector witness, LeftBound94791.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102256

namespace LeftBound102257
def owner : Owner := ⟨.program ⟨214⟩, ⟨30065⟩⟩
def transferEvent : Nat := 102257
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 102253 .summary, .result 94795 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102253 .summary)
      LeftBound102252.bound (LeftBound102252.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29788⟩⟩) (rawTerms := some (Proof.Events399.exact102253RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102252.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94795 .summary)
      LeftBound94794.bound (LeftBound94794.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30064⟩⟩) (rawTerms := some (Proof.Events370.exact94795RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94794.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102252.bound, LeftBound94794.bound]
def bound : CoeffClass := .finite ⟨23259036732736711122944, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102252.bound, LeftBound94794.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102252.actual selector witness, LeftBound94794.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound102257

namespace LeftBound102261
def owner : Owner := ⟨.program ⟨214⟩, ⟨30066⟩⟩
def transferEvent : Nat := 102261
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 102259 .coefficient) (.predecessor 1 102260 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102259 .coefficient)
      LeftBound102256.bound (LeftBound102256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102258RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102256.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102256.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102260 .coefficient)
      LeftAuthority94349.bound (LeftAuthority94349.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94350RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94349.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94349.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound102256.bound LeftAuthority94349.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102256.bound, LeftAuthority94349.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound102256.actual selector witness) * (LeftAuthority94349.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound102261

namespace LeftBound102262
def owner : Owner := ⟨.program ⟨214⟩, ⟨30066⟩⟩
def transferEvent : Nat := 102262
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩ [⟨.result 94350 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94350 .coefficient)
      LeftAuthority94349.bound (LeftAuthority94349.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18674⟩⟩) (rawTerms := some (Proof.Events368.exact94350RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94349.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94349.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority94349.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94349.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority94349.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound102262

namespace LeftBound102263
def owner : Owner := ⟨.program ⟨214⟩, ⟨30066⟩⟩
def transferEvent : Nat := 102263
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 102258 .summary) (.transfer 102262) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102258 .summary)
      LeftBound102257.bound (LeftBound102257.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30065⟩⟩) (rawTerms := some (Proof.Events399.exact102258RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102257.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 102262)
      LeftBound102262.bound (LeftBound102262.actual selector witness) := by
  exact .transfer (LeftBound102262.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound102257.bound LeftBound102262.bound
def bound : CoeffClass := .finite ⟨85361036953731453608582447104, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102257.bound, LeftBound102262.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound102257.actual selector witness) * (LeftBound102262.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound102263

namespace LeftBound102342
def owner : Owner := ⟨.program ⟨214⟩, ⟨18550⟩⟩
def transferEvent : Nat := 102342
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 102340 .coefficient) (.value (.predecessor 1 102341 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102340 .coefficient)
      LeftAuthority102338.bound (LeftAuthority102338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102339RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102338.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102338.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102341 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority102338.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority102338.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority102338.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound102342

namespace LeftBound102346
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def transferEvent : Nat := 102346
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 102344 .coefficient) (.predecessor 1 102345 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 102344 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 102345 .coefficient)
      LeftBound102342.bound (LeftBound102342.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102342.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102342.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound102342.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound102342.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound102342.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound102346

namespace LeftBound102347
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def transferEvent : Nat := 102347
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩ [⟨.result 102339 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102339 .coefficient)
      LeftAuthority102338.bound (LeftAuthority102338.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18548⟩⟩) (rawTerms := some (Proof.Events399.exact102339RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102338.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102338.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority102338.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority102338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority102338.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound102347

namespace LeftBound102348
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def transferEvent : Nat := 102348
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 102347) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 102347)
      LeftBound102347.bound (LeftBound102347.actual selector witness) := by
  exact .transfer (LeftBound102347.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound102347.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound102347.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound102347.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound102348

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
