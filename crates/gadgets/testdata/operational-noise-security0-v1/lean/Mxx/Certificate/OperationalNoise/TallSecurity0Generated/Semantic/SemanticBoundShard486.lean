import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard484
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard485

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound71336
def owner : Owner := ⟨.program ⟨214⟩, ⟨15701⟩⟩
def transferEvent : Nat := 71336
def frameStart : Nat := 71226
def rule : BoundRule := .sum [.predecessor 0 71334 .coefficient, .predecessor 1 71335 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71334 .coefficient)
      LeftAuthority71332.bound (LeftAuthority71332.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71333RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71332.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71332.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71335 .coefficient)
      LeftBound71328.bound (LeftBound71328.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71330RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71328.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71328.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority71332.bound, LeftBound71328.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71332.bound, LeftBound71328.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority71332.actual selector witness, LeftBound71328.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71336

namespace LeftBound71340
def owner : Owner := ⟨.program ⟨214⟩, ⟨25911⟩⟩
def transferEvent : Nat := 71340
def frameStart : Nat := 71226
def rule : BoundRule := .sum [.predecessor 0 71338 .coefficient, .predecessor 1 71339 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71338 .coefficient)
      LeftBound71336.bound (LeftBound71336.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71337RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71336.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71336.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71339 .coefficient)
      LeftBound71317.bound (LeftBound71317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71322RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71317.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71317.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71336.bound, LeftBound71317.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71336.bound, LeftBound71317.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71336.actual selector witness, LeftBound71317.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71340

namespace LeftBound71353
def owner : Owner := ⟨.program ⟨214⟩, ⟨25909⟩⟩
def transferEvent : Nat := 71353
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 71351 .coefficient, .predecessor 1 71352 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71351 .coefficient)
      LeftBound71174.bound (LeftBound71174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71350RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71174.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71174.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71352 .coefficient)
      LeftBound71157.bound (LeftBound71157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71164RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71157.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71157.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71174.bound, LeftBound71157.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71174.bound, LeftBound71157.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71174.actual selector witness, LeftBound71157.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71353

namespace LeftBound71356
def owner : Owner := ⟨.program ⟨214⟩, ⟨25909⟩⟩
def transferEvent : Nat := 71356
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 71350 .summary, .result 71164 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71350 .summary)
      LeftBound71176.bound (LeftBound71176.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19383⟩⟩) (rawTerms := some (Proof.Events278.exact71350RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound71176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71164 .summary)
      LeftBound71159.bound (LeftBound71159.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25908⟩⟩) (rawTerms := some (Proof.Events277.exact71164RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound71159.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71176.bound, LeftBound71159.bound]
def bound : CoeffClass := .finite ⟨352042398396416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71176.bound, LeftBound71159.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71176.actual selector witness, LeftBound71159.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71356

namespace LeftBound71360
def owner : Owner := ⟨.program ⟨214⟩, ⟨27421⟩⟩
def transferEvent : Nat := 71360
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 71358 .coefficient) (.predecessor 1 71359 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71358 .coefficient)
      LeftBound71353.bound (LeftBound71353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71357RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71353.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71353.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71359 .coefficient)
      LeftAuthority71079.bound (LeftAuthority71079.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71079.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71079.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound71353.bound LeftAuthority71079.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71353.bound, LeftAuthority71079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound71353.actual selector witness) * (LeftAuthority71079.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71360

namespace LeftBound71361
def owner : Owner := ⟨.program ⟨214⟩, ⟨27421⟩⟩
def transferEvent : Nat := 71361
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27419⟩⟩]⟩ [⟨.result 71080 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71080 .coefficient)
      LeftAuthority71079.bound (LeftAuthority71079.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27419⟩⟩) (rawTerms := some (Proof.Events277.exact71080RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71079.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71079.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority71079.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority71079.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound71361

namespace LeftBound71362
def owner : Owner := ⟨.program ⟨214⟩, ⟨27421⟩⟩
def transferEvent : Nat := 71362
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 71357 .summary) (.transfer 71361) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71357 .summary)
      LeftBound71356.bound (LeftBound71356.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25909⟩⟩) (rawTerms := some (Proof.Events278.exact71357RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound71356.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 71361)
      LeftBound71361.bound (LeftBound71361.actual selector witness) := by
  exact .transfer (LeftBound71361.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound71356.bound LeftBound71361.bound
def bound : CoeffClass := .finite ⟨1292001234793221062656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71356.bound, LeftBound71361.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound71356.actual selector witness) * (LeftBound71361.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71362

namespace LeftBound71373
def owner : Owner := ⟨.program ⟨214⟩, ⟨21110⟩⟩
def transferEvent : Nat := 71373
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 71371 .coefficient) (.value (.predecessor 1 71372 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71371 .coefficient)
      LeftAuthority71369.bound (LeftAuthority71369.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71369.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71369.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71372 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority71369.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71369.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority71369.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound71373

namespace LeftBound71377
def owner : Owner := ⟨.program ⟨214⟩, ⟨21111⟩⟩
def transferEvent : Nat := 71377
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 71375 .coefficient) (.predecessor 1 71376 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71375 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71376 .coefficient)
      LeftBound71373.bound (LeftBound71373.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events278.exact71374RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71373.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71373.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound71373.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound71373.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound71373.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71377

namespace LeftBound71378
def owner : Owner := ⟨.program ⟨214⟩, ⟨21111⟩⟩
def transferEvent : Nat := 71378
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21108⟩⟩]⟩ [⟨.result 71370 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71370 .coefficient)
      LeftAuthority71369.bound (LeftAuthority71369.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21108⟩⟩) (rawTerms := some (Proof.Events278.exact71370RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71369.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71369.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority71369.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71369.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority71369.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound71378

namespace LeftBound71379
def owner : Owner := ⟨.program ⟨214⟩, ⟨21111⟩⟩
def transferEvent : Nat := 71379
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 71378) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 71378)
      LeftBound71378.bound (LeftBound71378.actual selector witness) := by
  exact .transfer (LeftBound71378.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound71378.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound71378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound71378.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71379

namespace LeftBound71474
def owner : Owner := ⟨.program ⟨214⟩, ⟨15699⟩⟩
def transferEvent : Nat := 71474
def frameStart : Nat := 71435
def rule : BoundRule := .identity (.predecessor 0 71473 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71473 .coefficient)
      LeftAuthority71471.bound (LeftAuthority71471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events279.exact71472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71471.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71471.derived selector witness)

def rawBound : CoeffClass := LeftAuthority71471.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority71471.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound71474

namespace LeftBound71491
def owner : Owner := ⟨.program ⟨214⟩, ⟨15773⟩⟩
def transferEvent : Nat := 71491
def frameStart : Nat := 71435
def rule : BoundRule := .sum [.predecessor 0 71489 .coefficient, .predecessor 1 71490 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71489 .coefficient)
      LeftBound71474.bound (LeftBound71474.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound71474.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71490 .coefficient)
      LeftAuthority71487.bound (LeftAuthority71487.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority71487.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71474.bound, LeftAuthority71487.bound]
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71474.bound, LeftAuthority71487.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71474.actual selector witness, LeftAuthority71487.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71491

namespace LeftBound71494
def owner : Owner := ⟨.program ⟨214⟩, ⟨15774⟩⟩
def transferEvent : Nat := 71494
def frameStart : Nat := 71435
def rule : BoundRule := .identity (.predecessor 0 71493 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71493 .coefficient)
      LeftBound71491.bound (LeftBound71491.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound71491.derived selector witness)

def rawBound : CoeffClass := LeftBound71491.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71491.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound71491.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound71494

namespace LeftBound71500
def owner : Owner := ⟨.program ⟨214⟩, ⟨15775⟩⟩
def transferEvent : Nat := 71500
def frameStart : Nat := 71435
def rule : BoundRule := .product (.predecessor 0 71498 .coefficient) (.predecessor 1 71499 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71498 .coefficient)
      LeftAuthority71496.bound (LeftAuthority71496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events279.exact71497RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71496.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71496.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71499 .coefficient)
      LeftBound71494.bound (LeftBound71494.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events279.exact71495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71494.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71494.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority71496.bound LeftBound71494.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71496.bound, LeftBound71494.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority71496.actual selector witness) * (LeftBound71494.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71500

namespace LeftBound71508
def owner : Owner := ⟨.program ⟨214⟩, ⟨15776⟩⟩
def transferEvent : Nat := 71508
def frameStart : Nat := 71435
def rule : BoundRule := .sum [.predecessor 0 71506 .coefficient, .predecessor 1 71507 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71506 .coefficient)
      LeftAuthority71504.bound (LeftAuthority71504.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events279.exact71505RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71504.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71504.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71507 .coefficient)
      LeftBound71500.bound (LeftBound71500.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events279.exact71502RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71500.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71500.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority71504.bound, LeftBound71500.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71504.bound, LeftBound71500.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority71504.actual selector witness, LeftBound71500.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71508

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
