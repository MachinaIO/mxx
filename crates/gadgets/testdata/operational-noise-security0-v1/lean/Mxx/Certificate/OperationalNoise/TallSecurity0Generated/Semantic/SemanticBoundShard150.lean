import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard044
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard149

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound23295
def owner : Owner := ⟨.program ⟨214⟩, ⟨16723⟩⟩
def transferEvent : Nat := 23295
def frameStart : Nat := 23222
def rule : BoundRule := .sum [.predecessor 0 23293 .coefficient, .predecessor 1 23294 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23293 .coefficient)
      LeftAuthority23291.bound (LeftAuthority23291.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23292RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23291.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23291.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23294 .coefficient)
      LeftBound23287.bound (LeftBound23287.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23289RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23287.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23287.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority23291.bound, LeftBound23287.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23291.bound, LeftBound23287.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority23291.actual selector witness, LeftBound23287.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23295

namespace LeftBound23299
def owner : Owner := ⟨.program ⟨214⟩, ⟨29425⟩⟩
def transferEvent : Nat := 23299
def frameStart : Nat := 23222
def rule : BoundRule := .product (.predecessor 0 23297 .coefficient) (.predecessor 1 23298 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23297 .coefficient)
      LeftBound23295.bound (LeftBound23295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23295.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23298 .coefficient)
      LeftAuthority23272.bound (LeftAuthority23272.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23273RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23272.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23272.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound23295.bound LeftAuthority23272.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23295.bound, LeftAuthority23272.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound23295.actual selector witness) * (LeftAuthority23272.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23299

namespace LeftBound23310
def owner : Owner := ⟨.program ⟨214⟩, ⟨16689⟩⟩
def transferEvent : Nat := 23310
def frameStart : Nat := 23222
def rule : BoundRule := .product (.predecessor 0 23308 .coefficient) (.predecessor 1 23309 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23308 .coefficient)
      LeftAuthority23283.bound (LeftAuthority23283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23283.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23283.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23309 .coefficient)
      LeftAuthority23306.bound (LeftAuthority23306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23306.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23306.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority23283.bound LeftAuthority23306.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23283.bound, LeftAuthority23306.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority23283.actual selector witness) * (LeftAuthority23306.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23310

namespace LeftBound23318
def owner : Owner := ⟨.program ⟨214⟩, ⟨16690⟩⟩
def transferEvent : Nat := 23318
def frameStart : Nat := 23222
def rule : BoundRule := .sum [.predecessor 0 23316 .coefficient, .predecessor 1 23317 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23316 .coefficient)
      LeftAuthority23314.bound (LeftAuthority23314.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23315RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23314.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23314.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23317 .coefficient)
      LeftBound23310.bound (LeftBound23310.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23312RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23310.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23310.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority23314.bound, LeftBound23310.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23314.bound, LeftBound23310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority23314.actual selector witness, LeftBound23310.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23318

namespace LeftBound23322
def owner : Owner := ⟨.program ⟨214⟩, ⟨29429⟩⟩
def transferEvent : Nat := 23322
def frameStart : Nat := 23222
def rule : BoundRule := .sum [.predecessor 0 23320 .coefficient, .predecessor 1 23321 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23320 .coefficient)
      LeftBound23318.bound (LeftBound23318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23321 .coefficient)
      LeftBound23299.bound (LeftBound23299.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23304RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23299.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23299.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23318.bound, LeftBound23299.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23318.bound, LeftBound23299.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23318.actual selector witness, LeftBound23299.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23322

namespace LeftBound23335
def owner : Owner := ⟨.program ⟨214⟩, ⟨29427⟩⟩
def transferEvent : Nat := 23335
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 23333 .coefficient, .predecessor 1 23334 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23333 .coefficient)
      LeftBound23164.bound (LeftBound23164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23332RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23334 .coefficient)
      LeftBound23147.bound (LeftBound23147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23154RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23147.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23147.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23164.bound, LeftBound23147.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23164.bound, LeftBound23147.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23164.actual selector witness, LeftBound23147.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23335

namespace LeftBound23338
def owner : Owner := ⟨.program ⟨214⟩, ⟨29427⟩⟩
def transferEvent : Nat := 23338
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 23332 .summary, .result 23154 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23332 .summary)
      LeftBound23166.bound (LeftBound23166.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22423⟩⟩) (rawTerms := some (Proof.Events091.exact23332RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23166.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23154 .summary)
      LeftBound23149.bound (LeftBound23149.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29426⟩⟩) (rawTerms := some (Proof.Events090.exact23154RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23149.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23166.bound, LeftBound23149.bound]
def bound : CoeffClass := .finite ⟨1292382248169874534400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23166.bound, LeftBound23149.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23166.actual selector witness, LeftBound23149.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23338

namespace LeftBound23362
def owner : Owner := ⟨.program ⟨214⟩, ⟨12593⟩⟩
def transferEvent : Nat := 23362
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 23360 .coefficient) (.predecessor 1 23361 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23360 .coefficient)
      LeftAuthority933.bound (LeftAuthority933.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact934RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority933.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority933.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23361 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority933.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority933.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority933.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound23362

namespace LeftBound23367
def owner : Owner := ⟨.program ⟨214⟩, ⟨7356⟩⟩
def transferEvent : Nat := 23367
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 23365 .coefficient) (.predecessor 1 23366 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23365 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23366 .coefficient)
      LeftBound8475.bound (LeftBound8475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8475.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound8475.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound8475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound8475.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23367

namespace LeftBound23372
def owner : Owner := ⟨.program ⟨214⟩, ⟨12594⟩⟩
def transferEvent : Nat := 23372
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 23370 .coefficient, .predecessor 1 23371 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23370 .coefficient)
      LeftBound23367.bound (LeftBound23367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23371 .coefficient)
      LeftBound23362.bound (LeftBound23362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23362.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23362.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23367.bound, LeftBound23362.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23367.bound, LeftBound23362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23367.actual selector witness, LeftBound23362.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23372

namespace LeftBound23376
def owner : Owner := ⟨.program ⟨214⟩, ⟨12595⟩⟩
def transferEvent : Nat := 23376
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 23374 .coefficient, .predecessor 1 23375 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23374 .coefficient)
      LeftBound23372.bound (LeftBound23372.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23373RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23372.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23372.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23375 .coefficient)
      LeftBound8467.bound (LeftBound8467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8467.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23372.bound, LeftBound8467.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23372.bound, LeftBound8467.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23372.actual selector witness, LeftBound8467.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23376

namespace LeftBound23377
def owner : Owner := ⟨.program ⟨214⟩, ⟨12595⟩⟩
def transferEvent : Nat := 23377
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨100⟩⟩]⟩ [⟨.result 8468 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8468 .coefficient)
      LeftBound8467.bound (LeftBound8467.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨100⟩⟩) (rawTerms := some (Proof.Events033.exact8468RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8467.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound8467.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8467.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound8467.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound23377

namespace LeftBound23382
def owner : Owner := ⟨.program ⟨214⟩, ⟨12596⟩⟩
def transferEvent : Nat := 23382
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 23380 .coefficient) (.predecessor 1 23381 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23380 .coefficient)
      LeftBound23376.bound (LeftBound23376.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23379RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23376.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23376.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23381 .coefficient)
      LeftAuthority936.bound (LeftAuthority936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority936.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority936.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound23376.bound LeftAuthority936.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23376.bound, LeftAuthority936.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound23376.actual selector witness) * (LeftAuthority936.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23382

namespace LeftBound23383
def owner : Owner := ⟨.program ⟨214⟩, ⟨12596⟩⟩
def transferEvent : Nat := 23383
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9940⟩⟩], []⟩ [⟨.result 937 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 937 .coefficient)
      LeftAuthority936.bound (LeftAuthority936.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9940⟩⟩) (rawTerms := some (Proof.Events003.exact937RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority936.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority936.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority936.bound []
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority936.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority936.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound23383

namespace LeftBound23384
def owner : Owner := ⟨.program ⟨214⟩, ⟨12596⟩⟩
def transferEvent : Nat := 23384
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 23379 .summary) (.transfer 23383) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23379 .summary)
      LeftBound23377.bound (LeftBound23377.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12595⟩⟩) (rawTerms := some (Proof.Events091.exact23379RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23377.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 23383)
      LeftBound23383.bound (LeftBound23383.actual selector witness) := by
  exact .transfer (LeftBound23383.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound23377.bound LeftBound23383.bound
def bound : CoeffClass := .finite ⟨34944, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23377.bound, LeftBound23383.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound23377.actual selector witness) * (LeftBound23383.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23384

namespace LeftBound23390
def owner : Owner := ⟨.program ⟨214⟩, ⟨9941⟩⟩
def transferEvent : Nat := 23390
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 23388 .coefficient) (.predecessor 1 23389 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23388 .coefficient)
      LeftAuthority936.bound (LeftAuthority936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority936.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority936.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23389 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority936.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority936.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority936.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound23390

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
