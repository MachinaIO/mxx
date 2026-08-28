import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard017
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard022
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard023
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard024

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound6190
def owner : Owner := ⟨.program ⟨214⟩, ⟨6640⟩⟩
def transferEvent : Nat := 6190
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 6188 .coefficient) (.value (.predecessor 1 6189 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6188 .coefficient)
      LeftAuthority6186.bound (LeftAuthority6186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6187RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6186.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6186.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6189 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority6186.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6186.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6186.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound6190

namespace LeftBound6194
def owner : Owner := ⟨.program ⟨214⟩, ⟨7796⟩⟩
def transferEvent : Nat := 6194
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 6192 .coefficient) (.predecessor 1 6193 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6192 .coefficient)
      LeftBound5953.bound (LeftBound5953.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5954RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5953.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5953.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6193 .coefficient)
      LeftBound6190.bound (LeftBound6190.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6190.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6190.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound5953.bound LeftBound6190.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5953.bound, LeftBound6190.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound5953.actual selector witness) * (LeftBound6190.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6194

namespace LeftBound6217
def owner : Owner := ⟨.program ⟨214⟩, ⟨7797⟩⟩
def transferEvent : Nat := 6217
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6215 .coefficient, .predecessor 1 6216 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6215 .coefficient)
      LeftBound5876.bound (LeftBound5876.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5876.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5876.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6216 .coefficient)
      LeftBound6194.bound (LeftBound6194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6214RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6194.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6194.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound5876.bound, LeftBound6194.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5876.bound, LeftBound6194.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound5876.actual selector witness, LeftBound6194.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6217

namespace LeftBound6221
def owner : Owner := ⟨.program ⟨214⟩, ⟨7923⟩⟩
def transferEvent : Nat := 6221
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6219 .coefficient, .predecessor 1 6220 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6219 .coefficient)
      LeftBound6217.bound (LeftBound6217.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6217.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6217.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6220 .coefficient)
      LeftBound6177.bound (LeftBound6177.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6177.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6177.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6217.bound, LeftBound6177.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6217.bound, LeftBound6177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6217.actual selector witness, LeftBound6177.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6221

namespace LeftBound6225
def owner : Owner := ⟨.program ⟨214⟩, ⟨7924⟩⟩
def transferEvent : Nat := 6225
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6223 .coefficient, .predecessor 1 6224 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6223 .coefficient)
      LeftBound6221.bound (LeftBound6221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6222RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6221.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6221.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6224 .coefficient)
      LeftBound6137.bound (LeftBound6137.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6139RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6137.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6137.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6221.bound, LeftBound6137.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6221.bound, LeftBound6137.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6221.actual selector witness, LeftBound6137.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6225

namespace LeftBound6229
def owner : Owner := ⟨.program ⟨214⟩, ⟨7925⟩⟩
def transferEvent : Nat := 6229
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6227 .coefficient, .predecessor 1 6228 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6227 .coefficient)
      LeftBound6225.bound (LeftBound6225.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6226RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6225.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6225.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6228 .coefficient)
      LeftBound6097.bound (LeftBound6097.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6099RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6097.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6097.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6225.bound, LeftBound6097.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6225.bound, LeftBound6097.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6225.actual selector witness, LeftBound6097.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6229

namespace LeftBound6233
def owner : Owner := ⟨.program ⟨214⟩, ⟨7926⟩⟩
def transferEvent : Nat := 6233
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6231 .coefficient, .predecessor 1 6232 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6231 .coefficient)
      LeftBound6229.bound (LeftBound6229.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6230RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6229.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6229.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6232 .coefficient)
      LeftBound6057.bound (LeftBound6057.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6057.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6057.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6229.bound, LeftBound6057.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6229.bound, LeftBound6057.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6229.actual selector witness, LeftBound6057.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6233

namespace LeftBound6237
def owner : Owner := ⟨.program ⟨214⟩, ⟨7927⟩⟩
def transferEvent : Nat := 6237
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6235 .coefficient, .predecessor 1 6236 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6235 .coefficient)
      LeftBound6233.bound (LeftBound6233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6234RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6233.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6233.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6236 .coefficient)
      LeftBound6017.bound (LeftBound6017.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6019RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6017.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6017.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6233.bound, LeftBound6017.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6233.bound, LeftBound6017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6233.actual selector witness, LeftBound6017.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6237

namespace LeftBound6241
def owner : Owner := ⟨.program ⟨214⟩, ⟨7928⟩⟩
def transferEvent : Nat := 6241
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6239 .coefficient, .predecessor 1 6240 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6239 .coefficient)
      LeftBound6237.bound (LeftBound6237.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6238RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6237.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6237.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6240 .coefficient)
      LeftBound5977.bound (LeftBound5977.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5977.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5977.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6237.bound, LeftBound5977.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6237.bound, LeftBound5977.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6237.actual selector witness, LeftBound5977.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6241

namespace LeftBound6245
def owner : Owner := ⟨.program ⟨214⟩, ⟨7929⟩⟩
def transferEvent : Nat := 6245
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 6243 .coefficient) (.predecessor 1 6244 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6243 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6244 .coefficient)
      LeftBound6241.bound (LeftBound6241.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6242RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6241.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6241.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound6241.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound6241.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound6241.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6245

namespace LeftBound6274
def owner : Owner := ⟨.program ⟨214⟩, ⟨18909⟩⟩
def transferEvent : Nat := 6274
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6272 .coefficient, .predecessor 1 6273 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6272 .coefficient)
      LeftBound6245.bound (LeftBound6245.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6245.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6245.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6273 .coefficient)
      LeftBound5330.bound (LeftBound5330.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5464RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5330.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5330.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6245.bound, LeftBound5330.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6245.bound, LeftBound5330.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6245.actual selector witness, LeftBound5330.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6274

namespace LeftBound6297
def owner : Owner := ⟨.program ⟨214⟩, ⟨5619⟩⟩
def transferEvent : Nat := 6297
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 6292 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6292 .coefficient)
      LeftAuthority19.bound (LeftAuthority19.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact20RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19.derived selector witness)

def rawBound : CoeffClass := LeftAuthority19.bound
def bound : CoeffClass := .finite ⟨1, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority19.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound6297

namespace LeftBound6301
def owner : Owner := ⟨.program ⟨214⟩, ⟨6583⟩⟩
def transferEvent : Nat := 6301
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 6299 .coefficient) (.predecessor 1 6300 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6299 .coefficient)
      LeftBound6297.bound (LeftBound6297.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6298RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6297.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6297.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6300 .coefficient)
      LeftAuthority1.bound (LeftAuthority1.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact2RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound6297.bound LeftAuthority1.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6297.bound, LeftAuthority1.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound6297.actual selector witness) * (LeftAuthority1.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6301

namespace LeftBound6313
def owner : Owner := ⟨.program ⟨214⟩, ⟨5563⟩⟩
def transferEvent : Nat := 6313
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 6308 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6308 .coefficient)
      LeftAuthority19.bound (LeftAuthority19.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact20RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19.derived selector witness)

def rawBound : CoeffClass := LeftAuthority19.bound
def bound : CoeffClass := .finite ⟨1, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority19.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound6313

namespace LeftBound6317
def owner : Owner := ⟨.program ⟨214⟩, ⟨7365⟩⟩
def transferEvent : Nat := 6317
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 6315 .coefficient) (.predecessor 1 6316 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6315 .coefficient)
      LeftBound6313.bound (LeftBound6313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6316 .coefficient)
      LeftAuthority5479.bound (LeftAuthority5479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5479.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5479.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound6313.bound LeftAuthority5479.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6313.bound, LeftAuthority5479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound6313.actual selector witness) * (LeftAuthority5479.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6317

namespace LeftBound6322
def owner : Owner := ⟨.program ⟨214⟩, ⟨7767⟩⟩
def transferEvent : Nat := 6322
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6320 .coefficient, .predecessor 1 6321 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6320 .coefficient)
      LeftBound6317.bound (LeftBound6317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6317.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6317.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6321 .coefficient)
      LeftBound6301.bound (LeftBound6301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6303RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6301.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6301.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6317.bound, LeftBound6301.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6317.bound, LeftBound6301.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6317.actual selector witness, LeftBound6301.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6322

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
