import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard028
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard067

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound11434
def owner : Owner := ⟨.program ⟨214⟩, ⟨28140⟩⟩
def transferEvent : Nat := 11434
def frameStart : Nat := 11334
def rule : BoundRule := .sum [.predecessor 0 11432 .coefficient, .predecessor 1 11433 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11432 .coefficient)
      LeftBound11430.bound (LeftBound11430.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11431RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11430.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11430.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11433 .coefficient)
      LeftBound11411.bound (LeftBound11411.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11416RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11411.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11411.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11430.bound, LeftBound11411.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11430.bound, LeftBound11411.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11430.actual selector witness, LeftBound11411.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11434

namespace LeftBound11447
def owner : Owner := ⟨.program ⟨214⟩, ⟨28138⟩⟩
def transferEvent : Nat := 11447
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11445 .coefficient, .predecessor 1 11446 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11445 .coefficient)
      LeftBound11276.bound (LeftBound11276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11444RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11276.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11276.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11446 .coefficient)
      LeftBound11259.bound (LeftBound11259.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11266RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11259.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11259.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11276.bound, LeftBound11259.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11276.bound, LeftBound11259.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11276.actual selector witness, LeftBound11259.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11447

namespace LeftBound11450
def owner : Owner := ⟨.program ⟨214⟩, ⟨28138⟩⟩
def transferEvent : Nat := 11450
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 11444 .summary, .result 11266 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11444 .summary)
      LeftBound11278.bound (LeftBound11278.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21563⟩⟩) (rawTerms := some (Proof.Events044.exact11444RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound11278.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11266 .summary)
      LeftBound11261.bound (LeftBound11261.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28137⟩⟩) (rawTerms := some (Proof.Events044.exact11266RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound11261.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11278.bound, LeftBound11261.bound]
def bound : CoeffClass := .finite ⟨1292113298829627502592, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11278.bound, LeftBound11261.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11278.actual selector witness, LeftBound11261.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11450

namespace LeftBound11473
def owner : Owner := ⟨.program ⟨214⟩, ⟨93⟩⟩
def transferEvent : Nat := 11473
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 11472 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11472 .coefficient)
      LeftAuthority6440.bound (LeftAuthority6440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6440.derived selector witness)

def rawBound : CoeffClass := LeftAuthority6440.bound
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority6440.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound11473

namespace LeftBound11477
def owner : Owner := ⟨.program ⟨214⟩, ⟨11486⟩⟩
def transferEvent : Nat := 11477
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 11475 .coefficient) (.predecessor 1 11476 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11475 .coefficient)
      LeftAuthority280.bound (LeftAuthority280.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events001.exact281RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority280.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority280.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11476 .coefficient)
      LeftBound6447.bound (LeftBound6447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6447.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6447.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority280.bound LeftBound6447.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority280.bound, LeftBound6447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority280.actual selector witness) * (LeftBound6447.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound11477

namespace LeftBound11481
def owner : Owner := ⟨.program ⟨214⟩, ⟨6779⟩⟩
def transferEvent : Nat := 11481
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 11480 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11480 .coefficient)
      LeftAuthority5869.bound (LeftAuthority5869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5869.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5869.derived selector witness)

def rawBound : CoeffClass := LeftAuthority5869.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5869.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority5869.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound11481

namespace LeftBound11485
def owner : Owner := ⟨.program ⟨214⟩, ⟨7387⟩⟩
def transferEvent : Nat := 11485
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 11483 .coefficient) (.predecessor 1 11484 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11483 .coefficient)
      LeftBound6313.bound (LeftBound6313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11484 .coefficient)
      LeftBound11481.bound (LeftBound11481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11481.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11481.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound6313.bound LeftBound11481.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6313.bound, LeftBound11481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound6313.actual selector witness) * (LeftBound11481.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11485

namespace LeftBound11490
def owner : Owner := ⟨.program ⟨214⟩, ⟨11487⟩⟩
def transferEvent : Nat := 11490
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11488 .coefficient, .predecessor 1 11489 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11488 .coefficient)
      LeftBound11485.bound (LeftBound11485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11487RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11485.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11485.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11489 .coefficient)
      LeftBound11477.bound (LeftBound11477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11479RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11477.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11485.bound, LeftBound11477.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11485.bound, LeftBound11477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11485.actual selector witness, LeftBound11477.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11490

namespace LeftBound11494
def owner : Owner := ⟨.program ⟨214⟩, ⟨11488⟩⟩
def transferEvent : Nat := 11494
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11492 .coefficient, .predecessor 1 11493 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11492 .coefficient)
      LeftBound11490.bound (LeftBound11490.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11491RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11490.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11490.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11493 .coefficient)
      LeftBound11473.bound (LeftBound11473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11473.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11490.bound, LeftBound11473.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11490.bound, LeftBound11473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound11490.actual selector witness, LeftBound11473.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11494

namespace LeftBound11495
def owner : Owner := ⟨.program ⟨214⟩, ⟨11488⟩⟩
def transferEvent : Nat := 11495
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨93⟩⟩]⟩ [⟨.result 11474 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11474 .coefficient)
      LeftBound11473.bound (LeftBound11473.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨93⟩⟩) (rawTerms := some (Proof.Events044.exact11474RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11473.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound11473.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound11473.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound11495

namespace LeftBound11500
def owner : Owner := ⟨.program ⟨214⟩, ⟨14246⟩⟩
def transferEvent : Nat := 11500
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 11498 .coefficient) (.predecessor 1 11499 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11498 .coefficient)
      LeftBound11494.bound (LeftBound11494.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11497RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11494.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11494.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11499 .coefficient)
      LeftAuthority283.bound (LeftAuthority283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events001.exact284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound11494.bound LeftAuthority283.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11494.bound, LeftAuthority283.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound11494.actual selector witness) * (LeftAuthority283.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11500

namespace LeftBound11501
def owner : Owner := ⟨.program ⟨214⟩, ⟨14246⟩⟩
def transferEvent : Nat := 11501
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨14243⟩⟩], []⟩ [⟨.result 284 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 284 .coefficient)
      LeftAuthority283.bound (LeftAuthority283.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨14243⟩⟩) (rawTerms := some (Proof.Events001.exact284RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority283.bound []
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority283.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority283.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound11501

namespace LeftBound11502
def owner : Owner := ⟨.program ⟨214⟩, ⟨14246⟩⟩
def transferEvent : Nat := 11502
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 11497 .summary) (.transfer 11501) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11497 .summary)
      LeftBound11495.bound (LeftBound11495.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11488⟩⟩) (rawTerms := some (Proof.Events044.exact11497RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound11495.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 11501)
      LeftBound11501.bound (LeftBound11501.actual selector witness) := by
  exact .transfer (LeftBound11501.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound11495.bound LeftBound11501.bound
def bound : CoeffClass := .finite ⟨14976, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11495.bound, LeftBound11501.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound11495.actual selector witness) * (LeftBound11501.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11502

namespace LeftBound11511
def owner : Owner := ⟨.program ⟨214⟩, ⟨7853⟩⟩
def transferEvent : Nat := 11511
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 11509 .coefficient) (.value (.predecessor 1 11510 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11509 .coefficient)
      LeftAuthority11507.bound (LeftAuthority11507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11507.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11507.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11510 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority11507.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11507.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority11507.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound11511

namespace LeftBound11514
def owner : Owner := ⟨.program ⟨214⟩, ⟨73⟩⟩
def transferEvent : Nat := 11514
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 11513 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11513 .coefficient)
      LeftAuthority6440.bound (LeftAuthority6440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6440.derived selector witness)

def rawBound : CoeffClass := LeftAuthority6440.bound
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority6440.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound11514

namespace LeftBound11518
def owner : Owner := ⟨.program ⟨214⟩, ⟨14247⟩⟩
def transferEvent : Nat := 11518
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 11516 .coefficient) (.predecessor 1 11517 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 11516 .coefficient)
      LeftAuthority283.bound (LeftAuthority283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events001.exact284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 11517 .coefficient)
      LeftBound6447.bound (LeftBound6447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6447.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6447.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority283.bound LeftBound6447.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority283.bound, LeftBound6447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority283.actual selector witness) * (LeftBound6447.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound11518

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
