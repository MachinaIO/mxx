import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard091
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard092
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard396

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound58410
def owner : Owner := ⟨.program ⟨214⟩, ⟨10689⟩⟩
def transferEvent : Nat := 58410
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 58408 .coefficient, .predecessor 1 58409 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58408 .coefficient)
      LeftBound58406.bound (LeftBound58406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events228.exact58407RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58406.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58406.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58409 .coefficient)
      LeftBound14479.bound (LeftBound14479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14479.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58406.bound, LeftBound14479.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58406.bound, LeftBound14479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58406.actual selector witness, LeftBound14479.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58410

namespace LeftBound58411
def owner : Owner := ⟨.program ⟨214⟩, ⟨10689⟩⟩
def transferEvent : Nat := 58411
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨87⟩⟩]⟩ [⟨.result 14480 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14480 .coefficient)
      LeftBound14479.bound (LeftBound14479.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨87⟩⟩) (rawTerms := some (Proof.Events056.exact14480RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14479.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound14479.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound14479.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound58411

namespace LeftBound58416
def owner : Owner := ⟨.program ⟨214⟩, ⟨10690⟩⟩
def transferEvent : Nat := 58416
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 58414 .coefficient) (.predecessor 1 58415 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58414 .coefficient)
      LeftBound58410.bound (LeftBound58410.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events228.exact58413RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58410.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58410.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58415 .coefficient)
      LeftAuthority2708.bound (LeftAuthority2708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events010.exact2709RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2708.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2708.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound58410.bound LeftAuthority2708.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58410.bound, LeftAuthority2708.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound58410.actual selector witness) * (LeftAuthority2708.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58416

namespace LeftBound58417
def owner : Owner := ⟨.program ⟨214⟩, ⟨10690⟩⟩
def transferEvent : Nat := 58417
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9510⟩⟩], []⟩ [⟨.result 2709 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2709 .coefficient)
      LeftAuthority2708.bound (LeftAuthority2708.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9510⟩⟩) (rawTerms := some (Proof.Events010.exact2709RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2708.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2708.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority2708.bound []
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2708.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority2708.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound58417

namespace LeftBound58418
def owner : Owner := ⟨.program ⟨214⟩, ⟨10690⟩⟩
def transferEvent : Nat := 58418
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 58413 .summary) (.transfer 58417) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58413 .summary)
      LeftBound58411.bound (LeftBound58411.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10689⟩⟩) (rawTerms := some (Proof.Events228.exact58413RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58411.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 58417)
      LeftBound58417.bound (LeftBound58417.actual selector witness) := by
  exact .transfer (LeftBound58417.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound58411.bound LeftBound58417.bound
def bound : CoeffClass := .finite ⟨2496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58411.bound, LeftBound58417.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound58411.actual selector witness) * (LeftBound58417.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58418

namespace LeftBound58424
def owner : Owner := ⟨.program ⟨214⟩, ⟨9511⟩⟩
def transferEvent : Nat := 58424
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 58422 .coefficient) (.predecessor 1 58423 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58422 .coefficient)
      LeftAuthority2708.bound (LeftAuthority2708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events010.exact2709RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2708.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2708.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58423 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2708.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2708.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2708.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound58424

namespace LeftBound58429
def owner : Owner := ⟨.program ⟨214⟩, ⟨7276⟩⟩
def transferEvent : Nat := 58429
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 58427 .coefficient) (.predecessor 1 58428 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58427 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58428 .coefficient)
      LeftBound14528.bound (LeftBound14528.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14529RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14528.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14528.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound14528.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound14528.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound14528.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58429

namespace LeftBound58434
def owner : Owner := ⟨.program ⟨214⟩, ⟨9512⟩⟩
def transferEvent : Nat := 58434
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 58432 .coefficient, .predecessor 1 58433 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58432 .coefficient)
      LeftBound58429.bound (LeftBound58429.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events228.exact58431RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58429.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58429.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58433 .coefficient)
      LeftBound58424.bound (LeftBound58424.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events228.exact58426RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58424.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58424.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58429.bound, LeftBound58424.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58429.bound, LeftBound58424.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58429.actual selector witness, LeftBound58424.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58434

namespace LeftBound58438
def owner : Owner := ⟨.program ⟨214⟩, ⟨9513⟩⟩
def transferEvent : Nat := 58438
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 58436 .coefficient, .predecessor 1 58437 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58436 .coefficient)
      LeftBound58434.bound (LeftBound58434.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events228.exact58435RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58434.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58434.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58437 .coefficient)
      LeftBound14520.bound (LeftBound14520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14520.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58434.bound, LeftBound14520.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58434.bound, LeftBound14520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58434.actual selector witness, LeftBound14520.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58438

namespace LeftBound58439
def owner : Owner := ⟨.program ⟨214⟩, ⟨9513⟩⟩
def transferEvent : Nat := 58439
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨96⟩⟩]⟩ [⟨.result 14521 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14521 .coefficient)
      LeftBound14520.bound (LeftBound14520.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨96⟩⟩) (rawTerms := some (Proof.Events056.exact14521RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14520.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound14520.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound14520.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound58439

namespace LeftBound58444
def owner : Owner := ⟨.program ⟨214⟩, ⟨9514⟩⟩
def transferEvent : Nat := 58444
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 58442 .coefficient) (.predecessor 1 58443 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58442 .coefficient)
      LeftBound58438.bound (LeftBound58438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events228.exact58441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58438.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58438.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58443 .coefficient)
      LeftBound14517.bound (LeftBound14517.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14517.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14517.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound58438.bound LeftBound14517.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58438.bound, LeftBound14517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound58438.actual selector witness) * (LeftBound14517.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58444

namespace LeftBound58445
def owner : Owner := ⟨.program ⟨214⟩, ⟨9514⟩⟩
def transferEvent : Nat := 58445
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩ [⟨.result 14514 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14514 .coefficient)
      LeftAuthority14513.bound (LeftAuthority14513.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7834⟩⟩) (rawTerms := some (Proof.Events056.exact14514RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14513.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14513.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority14513.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14513.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority14513.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound58445

namespace LeftBound58446
def owner : Owner := ⟨.program ⟨214⟩, ⟨9514⟩⟩
def transferEvent : Nat := 58446
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 58441 .summary) (.transfer 58445) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58441 .summary)
      LeftBound58439.bound (LeftBound58439.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9513⟩⟩) (rawTerms := some (Proof.Events228.exact58441RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58439.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 58445)
      LeftBound58445.bound (LeftBound58445.actual selector witness) := by
  exact .transfer (LeftBound58445.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound58439.bound LeftBound58445.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58439.bound, LeftBound58445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound58439.actual selector witness) * (LeftBound58445.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58446

namespace LeftBound58454
def owner : Owner := ⟨.program ⟨214⟩, ⟨10691⟩⟩
def transferEvent : Nat := 58454
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 58452 .coefficient, .predecessor 1 58453 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58452 .coefficient)
      LeftBound58444.bound (LeftBound58444.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events228.exact58451RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58444.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58444.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58453 .coefficient)
      LeftBound58416.bound (LeftBound58416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events228.exact58421RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58416.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58416.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58444.bound, LeftBound58416.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58444.bound, LeftBound58416.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58444.actual selector witness, LeftBound58416.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58454

namespace LeftBound58456
def owner : Owner := ⟨.program ⟨214⟩, ⟨10691⟩⟩
def transferEvent : Nat := 58456
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 58451 .summary, .result 58421 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58451 .summary)
      LeftBound58446.bound (LeftBound58446.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9514⟩⟩) (rawTerms := some (Proof.Events228.exact58451RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58446.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58421 .summary)
      LeftBound58418.bound (LeftBound58418.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10690⟩⟩) (rawTerms := some (Proof.Events228.exact58421RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58418.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58446.bound, LeftBound58418.bound]
def bound : CoeffClass := .finite ⟨95422912, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58446.bound, LeftBound58418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58446.actual selector witness, LeftBound58418.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58456

namespace LeftBound58460
def owner : Owner := ⟨.program ⟨214⟩, ⟨24994⟩⟩
def transferEvent : Nat := 58460
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 58458 .coefficient) (.predecessor 1 58459 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58458 .coefficient)
      LeftBound58454.bound (LeftBound58454.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events228.exact58457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58454.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58454.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58459 .coefficient)
      LeftAuthority58392.bound (LeftAuthority58392.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events228.exact58393RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58392.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58392.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound58454.bound LeftAuthority58392.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58454.bound, LeftAuthority58392.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound58454.actual selector witness) * (LeftAuthority58392.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58460

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
