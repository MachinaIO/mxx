import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard344

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound51434
def owner : Owner := ⟨.program ⟨214⟩, ⟨29834⟩⟩
def transferEvent : Nat := 51434
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29832⟩⟩]⟩ [⟨.result 51153 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51153 .coefficient)
      LeftAuthority51152.bound (LeftAuthority51152.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29832⟩⟩) (rawTerms := some (Proof.Events199.exact51153RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51152.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51152.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority51152.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51152.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority51152.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound51434

namespace LeftBound51435
def owner : Owner := ⟨.program ⟨214⟩, ⟨29834⟩⟩
def transferEvent : Nat := 51435
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 51430 .summary) (.transfer 51434) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51430 .summary)
      LeftBound51429.bound (LeftBound51429.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25688⟩⟩) (rawTerms := some (Proof.Events200.exact51430RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51429.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 51434)
      LeftBound51434.bound (LeftBound51434.actual selector witness) := by
  exact .transfer (LeftBound51434.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51429.bound LeftBound51434.bound
def bound : CoeffClass := .finite ⟨1292516721028694540288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51429.bound, LeftBound51434.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51429.actual selector witness) * (LeftBound51434.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51435

namespace LeftBound51446
def owner : Owner := ⟨.program ⟨214⟩, ⟨22702⟩⟩
def transferEvent : Nat := 51446
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 51444 .coefficient) (.value (.predecessor 1 51445 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51444 .coefficient)
      LeftAuthority51442.bound (LeftAuthority51442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51443RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51442.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51442.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51445 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority51442.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51442.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority51442.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound51446

namespace LeftBound51450
def owner : Owner := ⟨.program ⟨214⟩, ⟨22703⟩⟩
def transferEvent : Nat := 51450
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 51448 .coefficient) (.predecessor 1 51449 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51448 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51449 .coefficient)
      LeftBound51446.bound (LeftBound51446.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51447RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51446.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51446.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound51446.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound51446.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound51446.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51450

namespace LeftBound51451
def owner : Owner := ⟨.program ⟨214⟩, ⟨22703⟩⟩
def transferEvent : Nat := 51451
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22700⟩⟩]⟩ [⟨.result 51443 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51443 .coefficient)
      LeftAuthority51442.bound (LeftAuthority51442.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22700⟩⟩) (rawTerms := some (Proof.Events200.exact51443RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51442.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51442.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority51442.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51442.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority51442.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound51451

namespace LeftBound51452
def owner : Owner := ⟨.program ⟨214⟩, ⟨22703⟩⟩
def transferEvent : Nat := 51452
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 51451) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 51451)
      LeftBound51451.bound (LeftBound51451.actual selector witness) := by
  exact .transfer (LeftBound51451.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound51451.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound51451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound51451.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51452

namespace LeftBound51547
def owner : Owner := ⟨.program ⟨214⟩, ⟨16876⟩⟩
def transferEvent : Nat := 51547
def frameStart : Nat := 51508
def rule : BoundRule := .identity (.predecessor 0 51546 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51546 .coefficient)
      LeftAuthority51544.bound (LeftAuthority51544.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51545RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51544.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51544.derived selector witness)

def rawBound : CoeffClass := LeftAuthority51544.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51544.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority51544.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound51547

namespace LeftBound51564
def owner : Owner := ⟨.program ⟨214⟩, ⟨16971⟩⟩
def transferEvent : Nat := 51564
def frameStart : Nat := 51508
def rule : BoundRule := .sum [.predecessor 0 51562 .coefficient, .predecessor 1 51563 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51562 .coefficient)
      LeftBound51547.bound (LeftBound51547.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound51547.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51563 .coefficient)
      LeftAuthority51560.bound (LeftAuthority51560.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority51560.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51547.bound, LeftAuthority51560.bound]
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51547.bound, LeftAuthority51560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51547.actual selector witness, LeftAuthority51560.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51564

namespace LeftBound51567
def owner : Owner := ⟨.program ⟨214⟩, ⟨16972⟩⟩
def transferEvent : Nat := 51567
def frameStart : Nat := 51508
def rule : BoundRule := .identity (.predecessor 0 51566 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51566 .coefficient)
      LeftBound51564.bound (LeftBound51564.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound51564.derived selector witness)

def rawBound : CoeffClass := LeftBound51564.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51564.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound51564.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound51567

namespace LeftBound51573
def owner : Owner := ⟨.program ⟨214⟩, ⟨16973⟩⟩
def transferEvent : Nat := 51573
def frameStart : Nat := 51508
def rule : BoundRule := .product (.predecessor 0 51571 .coefficient) (.predecessor 1 51572 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51571 .coefficient)
      LeftAuthority51569.bound (LeftAuthority51569.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51569.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51569.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51572 .coefficient)
      LeftBound51567.bound (LeftBound51567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51567.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51567.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority51569.bound LeftBound51567.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51569.bound, LeftBound51567.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority51569.actual selector witness) * (LeftBound51567.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51573

namespace LeftBound51581
def owner : Owner := ⟨.program ⟨214⟩, ⟨16974⟩⟩
def transferEvent : Nat := 51581
def frameStart : Nat := 51508
def rule : BoundRule := .sum [.predecessor 0 51579 .coefficient, .predecessor 1 51580 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51579 .coefficient)
      LeftAuthority51577.bound (LeftAuthority51577.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51578RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51577.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51577.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51580 .coefficient)
      LeftBound51573.bound (LeftBound51573.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51575RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51573.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51573.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority51577.bound, LeftBound51573.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51577.bound, LeftBound51573.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority51577.actual selector witness, LeftBound51573.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51581

namespace LeftBound51585
def owner : Owner := ⟨.program ⟨214⟩, ⟨29833⟩⟩
def transferEvent : Nat := 51585
def frameStart : Nat := 51508
def rule : BoundRule := .product (.predecessor 0 51583 .coefficient) (.predecessor 1 51584 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51583 .coefficient)
      LeftBound51581.bound (LeftBound51581.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51582RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51581.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51581.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51584 .coefficient)
      LeftAuthority51558.bound (LeftAuthority51558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51559RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51558.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51558.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51581.bound LeftAuthority51558.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51581.bound, LeftAuthority51558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51581.actual selector witness) * (LeftAuthority51558.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51585

namespace LeftBound51596
def owner : Owner := ⟨.program ⟨214⟩, ⟨17089⟩⟩
def transferEvent : Nat := 51596
def frameStart : Nat := 51508
def rule : BoundRule := .product (.predecessor 0 51594 .coefficient) (.predecessor 1 51595 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51594 .coefficient)
      LeftAuthority51569.bound (LeftAuthority51569.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51569.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51569.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51595 .coefficient)
      LeftAuthority51592.bound (LeftAuthority51592.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51592.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51592.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority51569.bound LeftAuthority51592.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51569.bound, LeftAuthority51592.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority51569.actual selector witness) * (LeftAuthority51592.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51596

namespace LeftBound51604
def owner : Owner := ⟨.program ⟨214⟩, ⟨17090⟩⟩
def transferEvent : Nat := 51604
def frameStart : Nat := 51508
def rule : BoundRule := .sum [.predecessor 0 51602 .coefficient, .predecessor 1 51603 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51602 .coefficient)
      LeftAuthority51600.bound (LeftAuthority51600.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51601RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51600.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51600.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51603 .coefficient)
      LeftBound51596.bound (LeftBound51596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51598RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51596.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51596.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority51600.bound, LeftBound51596.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51600.bound, LeftBound51596.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority51600.actual selector witness, LeftBound51596.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51604

namespace LeftBound51608
def owner : Owner := ⟨.program ⟨214⟩, ⟨29837⟩⟩
def transferEvent : Nat := 51608
def frameStart : Nat := 51508
def rule : BoundRule := .sum [.predecessor 0 51606 .coefficient, .predecessor 1 51607 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51606 .coefficient)
      LeftBound51604.bound (LeftBound51604.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51605RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51604.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51604.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51607 .coefficient)
      LeftBound51585.bound (LeftBound51585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51590RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51585.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51585.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51604.bound, LeftBound51585.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51604.bound, LeftBound51585.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51604.actual selector witness, LeftBound51585.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51608

namespace LeftBound51621
def owner : Owner := ⟨.program ⟨214⟩, ⟨29835⟩⟩
def transferEvent : Nat := 51621
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 51619 .coefficient, .predecessor 1 51620 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51619 .coefficient)
      LeftBound51450.bound (LeftBound51450.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51618RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51450.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51450.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51620 .coefficient)
      LeftBound51433.bound (LeftBound51433.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51440RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51433.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51433.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51450.bound, LeftBound51433.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51450.bound, LeftBound51433.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51450.actual selector witness, LeftBound51433.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51621

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
