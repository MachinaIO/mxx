import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard074
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard118

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound19359
def owner : Owner := ⟨.program ⟨214⟩, ⟨16033⟩⟩
def transferEvent : Nat := 19359
def frameStart : Nat := 19294
def rule : BoundRule := .product (.predecessor 0 19357 .coefficient) (.predecessor 1 19358 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19357 .coefficient)
      LeftAuthority19355.bound (LeftAuthority19355.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19356RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19355.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19355.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19358 .coefficient)
      LeftBound19353.bound (LeftBound19353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19354RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19353.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19353.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority19355.bound LeftBound19353.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19355.bound, LeftBound19353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority19355.actual selector witness) * (LeftBound19353.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19359

namespace LeftBound19367
def owner : Owner := ⟨.program ⟨214⟩, ⟨16034⟩⟩
def transferEvent : Nat := 19367
def frameStart : Nat := 19294
def rule : BoundRule := .sum [.predecessor 0 19365 .coefficient, .predecessor 1 19366 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19365 .coefficient)
      LeftAuthority19363.bound (LeftAuthority19363.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19363.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19363.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19366 .coefficient)
      LeftBound19359.bound (LeftBound19359.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19359.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19359.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority19363.bound, LeftBound19359.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19363.bound, LeftBound19359.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority19363.actual selector witness, LeftBound19359.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19367

namespace LeftBound19371
def owner : Owner := ⟨.program ⟨214⟩, ⟨27912⟩⟩
def transferEvent : Nat := 19371
def frameStart : Nat := 19294
def rule : BoundRule := .product (.predecessor 0 19369 .coefficient) (.predecessor 1 19370 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19369 .coefficient)
      LeftBound19367.bound (LeftBound19367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19368RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19370 .coefficient)
      LeftAuthority19344.bound (LeftAuthority19344.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19345RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19344.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19344.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound19367.bound LeftAuthority19344.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19367.bound, LeftAuthority19344.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound19367.actual selector witness) * (LeftAuthority19344.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19371

namespace LeftBound19382
def owner : Owner := ⟨.program ⟨214⟩, ⟨17183⟩⟩
def transferEvent : Nat := 19382
def frameStart : Nat := 19294
def rule : BoundRule := .product (.predecessor 0 19380 .coefficient) (.predecessor 1 19381 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19380 .coefficient)
      LeftAuthority19355.bound (LeftAuthority19355.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19356RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19355.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19355.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19381 .coefficient)
      LeftAuthority19378.bound (LeftAuthority19378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19379RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19378.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19378.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority19355.bound LeftAuthority19378.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19355.bound, LeftAuthority19378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority19355.actual selector witness) * (LeftAuthority19378.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19382

namespace LeftBound19390
def owner : Owner := ⟨.program ⟨214⟩, ⟨17184⟩⟩
def transferEvent : Nat := 19390
def frameStart : Nat := 19294
def rule : BoundRule := .sum [.predecessor 0 19388 .coefficient, .predecessor 1 19389 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19388 .coefficient)
      LeftAuthority19386.bound (LeftAuthority19386.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19386.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19386.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19389 .coefficient)
      LeftBound19382.bound (LeftBound19382.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19384RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19382.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19382.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority19386.bound, LeftBound19382.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19386.bound, LeftBound19382.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority19386.actual selector witness, LeftBound19382.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19390

namespace LeftBound19394
def owner : Owner := ⟨.program ⟨214⟩, ⟨27917⟩⟩
def transferEvent : Nat := 19394
def frameStart : Nat := 19294
def rule : BoundRule := .sum [.predecessor 0 19392 .coefficient, .predecessor 1 19393 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19392 .coefficient)
      LeftBound19390.bound (LeftBound19390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19391RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19390.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19390.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19393 .coefficient)
      LeftBound19371.bound (LeftBound19371.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19376RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19371.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19371.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound19390.bound, LeftBound19371.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19390.bound, LeftBound19371.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound19390.actual selector witness, LeftBound19371.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19394

namespace LeftBound19407
def owner : Owner := ⟨.program ⟨214⟩, ⟨27914⟩⟩
def transferEvent : Nat := 19407
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 19405 .coefficient, .predecessor 1 19406 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19405 .coefficient)
      LeftBound19236.bound (LeftBound19236.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19404RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19236.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19236.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19406 .coefficient)
      LeftBound19219.bound (LeftBound19219.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19226RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19219.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19219.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound19236.bound, LeftBound19219.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19236.bound, LeftBound19219.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound19236.actual selector witness, LeftBound19219.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19407

namespace LeftBound19410
def owner : Owner := ⟨.program ⟨214⟩, ⟨27914⟩⟩
def transferEvent : Nat := 19410
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 19404 .summary, .result 19226 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19404 .summary)
      LeftBound19238.bound (LeftBound19238.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21347⟩⟩) (rawTerms := some (Proof.Events075.exact19404RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound19238.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19226 .summary)
      LeftBound19221.bound (LeftBound19221.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27913⟩⟩) (rawTerms := some (Proof.Events075.exact19226RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound19221.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound19238.bound, LeftBound19221.bound]
def bound : CoeffClass := .finite ⟨1292068473939586330624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19238.bound, LeftBound19221.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound19238.actual selector witness, LeftBound19221.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19410

namespace LeftBound19414
def owner : Owner := ⟨.program ⟨214⟩, ⟨27915⟩⟩
def transferEvent : Nat := 19414
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 19412 .coefficient) (.predecessor 1 19413 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19412 .coefficient)
      LeftBound19407.bound (LeftBound19407.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19411RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19407.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19407.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19413 .coefficient)
      LeftBound5718.bound (LeftBound5718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5718.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5718.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound19407.bound LeftBound5718.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19407.bound, LeftBound5718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound19407.actual selector witness) * (LeftBound5718.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19414

namespace LeftBound19415
def owner : Owner := ⟨.program ⟨214⟩, ⟨27915⟩⟩
def transferEvent : Nat := 19415
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩ [⟨.result 5715 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5715 .coefficient)
      LeftAuthority5714.bound (LeftAuthority5714.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6641⟩⟩) (rawTerms := some (Proof.Events022.exact5715RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5714.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5714.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5714.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5714.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5714.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound19415

namespace LeftBound19416
def owner : Owner := ⟨.program ⟨214⟩, ⟨27915⟩⟩
def transferEvent : Nat := 19416
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 19411 .summary) (.transfer 19415) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19411 .summary)
      LeftBound19410.bound (LeftBound19410.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27914⟩⟩) (rawTerms := some (Proof.Events075.exact19411RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound19410.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 19415)
      LeftBound19415.bound (LeftBound19415.actual selector witness) := by
  exact .transfer (LeftBound19415.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound19410.bound LeftBound19415.bound
def bound : CoeffClass := .finite ⟨4741911972453864866771369984, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19410.bound, LeftBound19415.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound19410.actual selector witness) * (LeftBound19415.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19416

namespace LeftBound19431
def owner : Owner := ⟨.program ⟨214⟩, ⟨27696⟩⟩
def transferEvent : Nat := 19431
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 19429 .coefficient) (.predecessor 1 19430 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19429 .coefficient)
      LeftBound12254.bound (LeftBound12254.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12258RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12254.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12254.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19430 .coefficient)
      LeftAuthority19427.bound (LeftAuthority19427.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19428RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19427.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19427.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound12254.bound LeftAuthority19427.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12254.bound, LeftAuthority19427.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound12254.actual selector witness) * (LeftAuthority19427.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19431

namespace LeftBound19432
def owner : Owner := ⟨.program ⟨214⟩, ⟨27696⟩⟩
def transferEvent : Nat := 19432
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27694⟩⟩]⟩ [⟨.result 19428 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19428 .coefficient)
      LeftAuthority19427.bound (LeftAuthority19427.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27694⟩⟩) (rawTerms := some (Proof.Events075.exact19428RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19427.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19427.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority19427.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19427.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority19427.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound19432

namespace LeftBound19433
def owner : Owner := ⟨.program ⟨214⟩, ⟨27696⟩⟩
def transferEvent : Nat := 19433
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 12258 .summary) (.transfer 19432) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12258 .summary)
      LeftBound12257.bound (LeftBound12257.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26011⟩⟩) (rawTerms := some (Proof.Events047.exact12258RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound12257.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 19432)
      LeftBound19432.bound (LeftBound19432.actual selector witness) := by
  exact .transfer (LeftBound19432.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound12257.bound LeftBound19432.bound
def bound : CoeffClass := .finite ⟨1292046059683262234624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12257.bound, LeftBound19432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound12257.actual selector witness) * (LeftBound19432.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19433

namespace LeftBound19444
def owner : Owner := ⟨.program ⟨214⟩, ⟨21202⟩⟩
def transferEvent : Nat := 19444
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 19442 .coefficient) (.value (.predecessor 1 19443 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19442 .coefficient)
      LeftAuthority19440.bound (LeftAuthority19440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19440.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19443 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority19440.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19440.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority19440.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound19444

namespace LeftBound19448
def owner : Owner := ⟨.program ⟨214⟩, ⟨21203⟩⟩
def transferEvent : Nat := 19448
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 19446 .coefficient) (.predecessor 1 19447 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19446 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19447 .coefficient)
      LeftBound19444.bound (LeftBound19444.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19445RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19444.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19444.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound19444.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound19444.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound19444.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19448

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
