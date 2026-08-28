import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard049
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard050

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound9223
def owner : Owner := ⟨.program ⟨214⟩, ⟨16483⟩⟩
def transferEvent : Nat := 9223
def frameStart : Nat := 9121
def rule : BoundRule := .product (.predecessor 0 9221 .coefficient) (.predecessor 1 9222 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9221 .coefficient)
      LeftAuthority9176.bound (LeftAuthority9176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9177RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9176.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9222 .coefficient)
      LeftAuthority9219.bound (LeftAuthority9219.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9220RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9219.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9219.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority9176.bound LeftAuthority9219.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9176.bound, LeftAuthority9219.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority9176.actual selector witness) * (LeftAuthority9219.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9223

namespace LeftBound9231
def owner : Owner := ⟨.program ⟨214⟩, ⟨16484⟩⟩
def transferEvent : Nat := 9231
def frameStart : Nat := 9121
def rule : BoundRule := .sum [.predecessor 0 9229 .coefficient, .predecessor 1 9230 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9229 .coefficient)
      LeftAuthority9227.bound (LeftAuthority9227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9228RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9227.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9227.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9230 .coefficient)
      LeftBound9223.bound (LeftBound9223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9223.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9223.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority9227.bound, LeftBound9223.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9227.bound, LeftBound9223.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority9227.actual selector witness, LeftBound9223.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9231

namespace LeftBound9235
def owner : Owner := ⟨.program ⟨214⟩, ⟨25397⟩⟩
def transferEvent : Nat := 9235
def frameStart : Nat := 9121
def rule : BoundRule := .sum [.predecessor 0 9233 .coefficient, .predecessor 1 9234 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9233 .coefficient)
      LeftBound9231.bound (LeftBound9231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9232RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9231.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9231.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9234 .coefficient)
      LeftBound9212.bound (LeftBound9212.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9217RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9212.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9212.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9231.bound, LeftBound9212.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9231.bound, LeftBound9212.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9231.actual selector witness, LeftBound9212.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9235

namespace LeftBound9248
def owner : Owner := ⟨.program ⟨214⟩, ⟨25395⟩⟩
def transferEvent : Nat := 9248
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9246 .coefficient, .predecessor 1 9247 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9246 .coefficient)
      LeftBound9069.bound (LeftBound9069.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9069.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9069.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9247 .coefficient)
      LeftBound9052.bound (LeftBound9052.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9052.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9052.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9069.bound, LeftBound9052.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9069.bound, LeftBound9052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9069.actual selector witness, LeftBound9052.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9248

namespace LeftBound9251
def owner : Owner := ⟨.program ⟨214⟩, ⟨25395⟩⟩
def transferEvent : Nat := 9251
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 9245 .summary, .result 9059 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9245 .summary)
      LeftBound9071.bound (LeftBound9071.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19907⟩⟩) (rawTerms := some (Proof.Events036.exact9245RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound9071.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9059 .summary)
      LeftBound9054.bound (LeftBound9054.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25394⟩⟩) (rawTerms := some (Proof.Events035.exact9059RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound9054.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9071.bound, LeftBound9054.bound]
def bound : CoeffClass := .finite ⟨352127895089152, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9071.bound, LeftBound9054.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9071.actual selector witness, LeftBound9054.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9251

namespace LeftBound9255
def owner : Owner := ⟨.program ⟨214⟩, ⟨29005⟩⟩
def transferEvent : Nat := 9255
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9253 .coefficient) (.predecessor 1 9254 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9253 .coefficient)
      LeftBound9248.bound (LeftBound9248.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9252RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9248.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9248.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9254 .coefficient)
      LeftAuthority8955.bound (LeftAuthority8955.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8956RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8955.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8955.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound9248.bound LeftAuthority8955.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9248.bound, LeftAuthority8955.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound9248.actual selector witness) * (LeftAuthority8955.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9255

namespace LeftBound9256
def owner : Owner := ⟨.program ⟨214⟩, ⟨29005⟩⟩
def transferEvent : Nat := 9256
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29003⟩⟩]⟩ [⟨.result 8956 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8956 .coefficient)
      LeftAuthority8955.bound (LeftAuthority8955.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29003⟩⟩) (rawTerms := some (Proof.Events034.exact8956RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8955.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8955.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority8955.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8955.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority8955.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound9256

namespace LeftBound9257
def owner : Owner := ⟨.program ⟨214⟩, ⟨29005⟩⟩
def transferEvent : Nat := 9257
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 9252 .summary) (.transfer 9256) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9252 .summary)
      LeftBound9251.bound (LeftBound9251.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25395⟩⟩) (rawTerms := some (Proof.Events036.exact9252RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound9251.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 9256)
      LeftBound9256.bound (LeftBound9256.actual selector witness) := by
  exact .transfer (LeftBound9256.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound9251.bound LeftBound9256.bound
def bound : CoeffClass := .finite ⟨1292315009023509266432, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9251.bound, LeftBound9256.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound9251.actual selector witness) * (LeftBound9256.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9257

namespace LeftBound9268
def owner : Owner := ⟨.program ⟨214⟩, ⟨22138⟩⟩
def transferEvent : Nat := 9268
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 9266 .coefficient) (.value (.predecessor 1 9267 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9266 .coefficient)
      LeftAuthority9264.bound (LeftAuthority9264.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9265RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9264.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9264.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9267 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority9264.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9264.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority9264.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound9268

namespace LeftBound9272
def owner : Owner := ⟨.program ⟨214⟩, ⟨22139⟩⟩
def transferEvent : Nat := 9272
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9270 .coefficient) (.predecessor 1 9271 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9270 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9271 .coefficient)
      LeftBound9268.bound (LeftBound9268.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9269RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9268.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9268.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound9268.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound9268.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound9268.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9272

namespace LeftBound9273
def owner : Owner := ⟨.program ⟨214⟩, ⟨22139⟩⟩
def transferEvent : Nat := 9273
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22136⟩⟩]⟩ [⟨.result 9265 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9265 .coefficient)
      LeftAuthority9264.bound (LeftAuthority9264.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22136⟩⟩) (rawTerms := some (Proof.Events036.exact9265RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9264.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9264.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority9264.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9264.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority9264.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound9273

namespace LeftBound9274
def owner : Owner := ⟨.program ⟨214⟩, ⟨22139⟩⟩
def transferEvent : Nat := 9274
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 9273) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 9273)
      LeftBound9273.bound (LeftBound9273.actual selector witness) := by
  exact .transfer (LeftBound9273.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound9273.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound9273.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound9273.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9274

namespace LeftBound9369
def owner : Owner := ⟨.program ⟨214⟩, ⟨16482⟩⟩
def transferEvent : Nat := 9369
def frameStart : Nat := 9330
def rule : BoundRule := .identity (.predecessor 0 9368 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9368 .coefficient)
      LeftAuthority9366.bound (LeftAuthority9366.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9367RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9366.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9366.derived selector witness)

def rawBound : CoeffClass := LeftAuthority9366.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9366.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority9366.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound9369

namespace LeftBound9386
def owner : Owner := ⟨.program ⟨214⟩, ⟨16521⟩⟩
def transferEvent : Nat := 9386
def frameStart : Nat := 9330
def rule : BoundRule := .sum [.predecessor 0 9384 .coefficient, .predecessor 1 9385 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9384 .coefficient)
      LeftBound9369.bound (LeftBound9369.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound9369.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9385 .coefficient)
      LeftAuthority9382.bound (LeftAuthority9382.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority9382.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9369.bound, LeftAuthority9382.bound]
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9369.bound, LeftAuthority9382.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9369.actual selector witness, LeftAuthority9382.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9386

namespace LeftBound9389
def owner : Owner := ⟨.program ⟨214⟩, ⟨16522⟩⟩
def transferEvent : Nat := 9389
def frameStart : Nat := 9330
def rule : BoundRule := .identity (.predecessor 0 9388 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9388 .coefficient)
      LeftBound9386.bound (LeftBound9386.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound9386.derived selector witness)

def rawBound : CoeffClass := LeftBound9386.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9386.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound9386.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound9389

namespace LeftBound9395
def owner : Owner := ⟨.program ⟨214⟩, ⟨16523⟩⟩
def transferEvent : Nat := 9395
def frameStart : Nat := 9330
def rule : BoundRule := .product (.predecessor 0 9393 .coefficient) (.predecessor 1 9394 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9393 .coefficient)
      LeftAuthority9391.bound (LeftAuthority9391.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9392RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9391.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9391.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9394 .coefficient)
      LeftBound9389.bound (LeftBound9389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9390RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9389.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9389.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority9391.bound LeftBound9389.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9391.bound, LeftBound9389.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority9391.actual selector witness) * (LeftBound9389.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9395

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
