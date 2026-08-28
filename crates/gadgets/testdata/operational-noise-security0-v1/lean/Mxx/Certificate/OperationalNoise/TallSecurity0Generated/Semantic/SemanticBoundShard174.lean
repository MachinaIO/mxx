import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard173

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound26446
def owner : Owner := ⟨.program ⟨214⟩, ⟨14328⟩⟩
def transferEvent : Nat := 26446
def frameStart : Nat := 26387
def rule : BoundRule := .product (.predecessor 0 26444 .coefficient) (.predecessor 1 26445 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26444 .coefficient)
      LeftAuthority26442.bound (LeftAuthority26442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26443RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26442.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26442.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26445 .coefficient)
      LeftBound26440.bound (LeftBound26440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26440.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26440.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority26442.bound LeftBound26440.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26442.bound, LeftBound26440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority26442.actual selector witness) * (LeftBound26440.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26446

namespace LeftBound26462
def owner : Owner := ⟨.program ⟨214⟩, ⟨7853⟩⟩
def transferEvent : Nat := 26462
def frameStart : Nat := 26387
def rule : BoundRule := .scale (.predecessor 0 26460 .coefficient) (.value (.predecessor 1 26461 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26460 .coefficient)
      LeftAuthority26458.bound (LeftAuthority26458.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26459RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26458.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26458.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26461 .coefficient)
      LeftAuthority26449.bound (LeftAuthority26449.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority26449.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority26458.bound LeftAuthority26449.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26458.bound, LeftAuthority26449.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority26458.actual selector witness) * (LeftAuthority26449.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound26462

namespace LeftBound26465
def owner : Owner := ⟨.program ⟨214⟩, ⟨6759⟩⟩
def transferEvent : Nat := 26465
def frameStart : Nat := 26387
def rule : BoundRule := .identity (.predecessor 0 26464 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26464 .coefficient)
      LeftAuthority26452.bound (LeftAuthority26452.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26453RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26452.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26452.derived selector witness)

def rawBound : CoeffClass := LeftAuthority26452.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26452.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority26452.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound26465

namespace LeftBound26469
def owner : Owner := ⟨.program ⟨214⟩, ⟨7854⟩⟩
def transferEvent : Nat := 26469
def frameStart : Nat := 26387
def rule : BoundRule := .product (.predecessor 0 26467 .coefficient) (.predecessor 1 26468 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26467 .coefficient)
      LeftBound26465.bound (LeftBound26465.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26466RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26465.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26468 .coefficient)
      LeftBound26462.bound (LeftBound26462.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26462.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26462.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound26465.bound LeftBound26462.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26465.bound, LeftBound26462.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound26465.actual selector witness) * (LeftBound26462.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26469

namespace LeftBound26474
def owner : Owner := ⟨.program ⟨214⟩, ⟨14329⟩⟩
def transferEvent : Nat := 26474
def frameStart : Nat := 26387
def rule : BoundRule := .sum [.predecessor 0 26472 .coefficient, .predecessor 1 26473 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26472 .coefficient)
      LeftBound26469.bound (LeftBound26469.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26471RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26469.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26469.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26473 .coefficient)
      LeftBound26446.bound (LeftBound26446.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26448RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26446.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26446.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26469.bound, LeftBound26446.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26469.bound, LeftBound26446.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26469.actual selector witness, LeftBound26446.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26474

namespace LeftBound26478
def owner : Owner := ⟨.program ⟨214⟩, ⟨26084⟩⟩
def transferEvent : Nat := 26478
def frameStart : Nat := 26387
def rule : BoundRule := .product (.predecessor 0 26476 .coefficient) (.predecessor 1 26477 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26476 .coefficient)
      LeftBound26474.bound (LeftBound26474.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26474.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26474.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26477 .coefficient)
      LeftAuthority26431.bound (LeftAuthority26431.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26432RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26431.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26431.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound26474.bound LeftAuthority26431.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26474.bound, LeftAuthority26431.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound26474.actual selector witness) * (LeftAuthority26431.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26478

namespace LeftBound26489
def owner : Owner := ⟨.program ⟨214⟩, ⟨15954⟩⟩
def transferEvent : Nat := 26489
def frameStart : Nat := 26387
def rule : BoundRule := .product (.predecessor 0 26487 .coefficient) (.predecessor 1 26488 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26487 .coefficient)
      LeftAuthority26442.bound (LeftAuthority26442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26443RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26442.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26442.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26488 .coefficient)
      LeftAuthority26485.bound (LeftAuthority26485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26485.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26485.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority26442.bound LeftAuthority26485.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26442.bound, LeftAuthority26485.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority26442.actual selector witness) * (LeftAuthority26485.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26489

namespace LeftBound26497
def owner : Owner := ⟨.program ⟨214⟩, ⟨15955⟩⟩
def transferEvent : Nat := 26497
def frameStart : Nat := 26387
def rule : BoundRule := .sum [.predecessor 0 26495 .coefficient, .predecessor 1 26496 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26495 .coefficient)
      LeftAuthority26493.bound (LeftAuthority26493.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26493.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26496 .coefficient)
      LeftBound26489.bound (LeftBound26489.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26491RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26489.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26489.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority26493.bound, LeftBound26489.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26493.bound, LeftBound26489.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority26493.actual selector witness, LeftBound26489.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26497

namespace LeftBound26501
def owner : Owner := ⟨.program ⟨214⟩, ⟨26085⟩⟩
def transferEvent : Nat := 26501
def frameStart : Nat := 26387
def rule : BoundRule := .sum [.predecessor 0 26499 .coefficient, .predecessor 1 26500 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26499 .coefficient)
      LeftBound26497.bound (LeftBound26497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26497.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26497.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26500 .coefficient)
      LeftBound26478.bound (LeftBound26478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26478.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26478.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26497.bound, LeftBound26478.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26497.bound, LeftBound26478.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26497.actual selector witness, LeftBound26478.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26501

namespace LeftBound26514
def owner : Owner := ⟨.program ⟨214⟩, ⟨26083⟩⟩
def transferEvent : Nat := 26514
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26512 .coefficient, .predecessor 1 26513 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26512 .coefficient)
      LeftBound26335.bound (LeftBound26335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26511RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26335.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26335.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26513 .coefficient)
      LeftBound26318.bound (LeftBound26318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26318.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26335.bound, LeftBound26318.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26335.bound, LeftBound26318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26335.actual selector witness, LeftBound26318.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26514

namespace LeftBound26517
def owner : Owner := ⟨.program ⟨214⟩, ⟨26083⟩⟩
def transferEvent : Nat := 26517
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26511 .summary, .result 26325 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26511 .summary)
      LeftBound26337.bound (LeftBound26337.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19543⟩⟩) (rawTerms := some (Proof.Events103.exact26511RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26325 .summary)
      LeftBound26320.bound (LeftBound26320.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26082⟩⟩) (rawTerms := some (Proof.Events102.exact26325RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26320.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26337.bound, LeftBound26320.bound]
def bound : CoeffClass := .finite ⟨352060719116288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26337.bound, LeftBound26320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26337.actual selector witness, LeftBound26320.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26517

namespace LeftBound26521
def owner : Owner := ⟨.program ⟨214⟩, ⟨27907⟩⟩
def transferEvent : Nat := 26521
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 26519 .coefficient) (.predecessor 1 26520 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26519 .coefficient)
      LeftBound26514.bound (LeftBound26514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26514.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26520 .coefficient)
      LeftAuthority26240.bound (LeftAuthority26240.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26241RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26240.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26240.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound26514.bound LeftAuthority26240.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26514.bound, LeftAuthority26240.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound26514.actual selector witness) * (LeftAuthority26240.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26521

namespace LeftBound26522
def owner : Owner := ⟨.program ⟨214⟩, ⟨27907⟩⟩
def transferEvent : Nat := 26522
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27905⟩⟩]⟩ [⟨.result 26241 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26241 .coefficient)
      LeftAuthority26240.bound (LeftAuthority26240.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27905⟩⟩) (rawTerms := some (Proof.Events102.exact26241RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26240.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26240.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority26240.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26240.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority26240.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound26522

namespace LeftBound26523
def owner : Owner := ⟨.program ⟨214⟩, ⟨27907⟩⟩
def transferEvent : Nat := 26523
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 26518 .summary) (.transfer 26522) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26518 .summary)
      LeftBound26517.bound (LeftBound26517.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26083⟩⟩) (rawTerms := some (Proof.Events103.exact26518RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26517.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 26522)
      LeftBound26522.bound (LeftBound26522.actual selector witness) := by
  exact .transfer (LeftBound26522.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound26517.bound LeftBound26522.bound
def bound : CoeffClass := .finite ⟨1292068472128282820608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26517.bound, LeftBound26522.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound26517.actual selector witness) * (LeftBound26522.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26523

namespace LeftBound26534
def owner : Owner := ⟨.program ⟨214⟩, ⟨21414⟩⟩
def transferEvent : Nat := 26534
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 26532 .coefficient) (.value (.predecessor 1 26533 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26532 .coefficient)
      LeftAuthority26530.bound (LeftAuthority26530.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26531RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26530.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26530.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26533 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority26530.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26530.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority26530.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound26534

namespace LeftBound26538
def owner : Owner := ⟨.program ⟨214⟩, ⟨21415⟩⟩
def transferEvent : Nat := 26538
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 26536 .coefficient) (.predecessor 1 26537 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26536 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26537 .coefficient)
      LeftBound26534.bound (LeftBound26534.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26535RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26534.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26534.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound26534.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound26534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound26534.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26538

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
