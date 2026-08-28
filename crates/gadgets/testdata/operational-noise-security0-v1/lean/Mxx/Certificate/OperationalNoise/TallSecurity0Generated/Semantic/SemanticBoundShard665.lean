import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard664

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound97341
def owner : Owner := ⟨.program ⟨214⟩, ⟨16414⟩⟩
def transferEvent : Nat := 97341
def frameStart : Nat := 97294
def rule : BoundRule := .identity (.predecessor 0 97340 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97340 .coefficient)
      LeftBound97338.bound (LeftBound97338.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound97338.derived selector witness)

def rawBound : CoeffClass := LeftBound97338.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound97338.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound97341

namespace LeftBound97347
def owner : Owner := ⟨.program ⟨214⟩, ⟨16415⟩⟩
def transferEvent : Nat := 97347
def frameStart : Nat := 97294
def rule : BoundRule := .product (.predecessor 0 97345 .coefficient) (.predecessor 1 97346 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97345 .coefficient)
      LeftAuthority97343.bound (LeftAuthority97343.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97344RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97343.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97343.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97346 .coefficient)
      LeftBound97341.bound (LeftBound97341.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97342RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97341.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97341.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority97343.bound LeftBound97341.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97343.bound, LeftBound97341.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority97343.actual selector witness) * (LeftBound97341.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97347

namespace LeftBound97355
def owner : Owner := ⟨.program ⟨214⟩, ⟨16416⟩⟩
def transferEvent : Nat := 97355
def frameStart : Nat := 97294
def rule : BoundRule := .sum [.predecessor 0 97353 .coefficient, .predecessor 1 97354 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97353 .coefficient)
      LeftAuthority97351.bound (LeftAuthority97351.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97352RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97351.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97351.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97354 .coefficient)
      LeftBound97347.bound (LeftBound97347.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97347.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97347.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority97351.bound, LeftBound97347.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97351.bound, LeftBound97347.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority97351.actual selector witness, LeftBound97347.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97355

namespace LeftBound97359
def owner : Owner := ⟨.program ⟨214⟩, ⟨28700⟩⟩
def transferEvent : Nat := 97359
def frameStart : Nat := 97294
def rule : BoundRule := .product (.predecessor 0 97357 .coefficient) (.predecessor 1 97358 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97357 .coefficient)
      LeftBound97355.bound (LeftBound97355.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97356RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97355.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97355.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97358 .coefficient)
      LeftAuthority97332.bound (LeftAuthority97332.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97333RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97332.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97332.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97355.bound LeftAuthority97332.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97355.bound, LeftAuthority97332.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97355.actual selector witness) * (LeftAuthority97332.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97359

namespace LeftBound97370
def owner : Owner := ⟨.program ⟨214⟩, ⟨17114⟩⟩
def transferEvent : Nat := 97370
def frameStart : Nat := 97294
def rule : BoundRule := .product (.predecessor 0 97368 .coefficient) (.predecessor 1 97369 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97368 .coefficient)
      LeftAuthority97343.bound (LeftAuthority97343.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97344RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97343.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97343.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97369 .coefficient)
      LeftAuthority97366.bound (LeftAuthority97366.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97367RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97366.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97366.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority97343.bound LeftAuthority97366.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97343.bound, LeftAuthority97366.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority97343.actual selector witness) * (LeftAuthority97366.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97370

namespace LeftBound97378
def owner : Owner := ⟨.program ⟨214⟩, ⟨17115⟩⟩
def transferEvent : Nat := 97378
def frameStart : Nat := 97294
def rule : BoundRule := .sum [.predecessor 0 97376 .coefficient, .predecessor 1 97377 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97376 .coefficient)
      LeftAuthority97374.bound (LeftAuthority97374.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97375RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97374.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97374.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97377 .coefficient)
      LeftBound97370.bound (LeftBound97370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97372RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97370.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97370.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority97374.bound, LeftBound97370.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97374.bound, LeftBound97370.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority97374.actual selector witness, LeftBound97370.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97378

namespace LeftBound97382
def owner : Owner := ⟨.program ⟨214⟩, ⟨28704⟩⟩
def transferEvent : Nat := 97382
def frameStart : Nat := 97294
def rule : BoundRule := .sum [.predecessor 0 97380 .coefficient, .predecessor 1 97381 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97380 .coefficient)
      LeftBound97378.bound (LeftBound97378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97379RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97378.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97378.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97381 .coefficient)
      LeftBound97359.bound (LeftBound97359.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97359.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97359.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97378.bound, LeftBound97359.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97378.bound, LeftBound97359.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97378.actual selector witness, LeftBound97359.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97382

namespace LeftBound97395
def owner : Owner := ⟨.program ⟨214⟩, ⟨28702⟩⟩
def transferEvent : Nat := 97395
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 97393 .coefficient, .predecessor 1 97394 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97393 .coefficient)
      LeftBound97248.bound (LeftBound97248.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97392RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97248.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97248.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97394 .coefficient)
      LeftBound97231.bound (LeftBound97231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97238RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97231.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97231.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97248.bound, LeftBound97231.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97248.bound, LeftBound97231.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97248.actual selector witness, LeftBound97231.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97395

namespace LeftBound97398
def owner : Owner := ⟨.program ⟨214⟩, ⟨28702⟩⟩
def transferEvent : Nat := 97398
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 97392 .summary, .result 97238 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97392 .summary)
      LeftBound97250.bound (LeftBound97250.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21968⟩⟩) (rawTerms := some (Proof.Events380.exact97392RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97250.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97238 .summary)
      LeftBound97233.bound (LeftBound97233.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28701⟩⟩) (rawTerms := some (Proof.Events379.exact97238RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97233.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97250.bound, LeftBound97233.bound]
def bound : CoeffClass := .finite ⟨1292270185944771604480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97250.bound, LeftBound97233.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97250.actual selector witness, LeftBound97233.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97398

namespace LeftBound97422
def owner : Owner := ⟨.program ⟨214⟩, ⟨11740⟩⟩
def transferEvent : Nat := 97422
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 97420 .coefficient) (.predecessor 1 97421 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97420 .coefficient)
      LeftAuthority4726.bound (LeftAuthority4726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4726.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97421 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4726.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4726.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4726.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound97422

namespace LeftBound97427
def owner : Owner := ⟨.program ⟨214⟩, ⟨7120⟩⟩
def transferEvent : Nat := 97427
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 97425 .coefficient) (.predecessor 1 97426 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97425 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97426 .coefficient)
      LeftBound9978.bound (LeftBound9978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9978.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound9978.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound9978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound9978.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97427

namespace LeftBound97432
def owner : Owner := ⟨.program ⟨214⟩, ⟨11741⟩⟩
def transferEvent : Nat := 97432
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 97430 .coefficient, .predecessor 1 97431 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97430 .coefficient)
      LeftBound97427.bound (LeftBound97427.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97429RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97427.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97427.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97431 .coefficient)
      LeftBound97422.bound (LeftBound97422.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97424RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97422.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97422.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97427.bound, LeftBound97422.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97427.bound, LeftBound97422.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97427.actual selector witness, LeftBound97422.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97432

namespace LeftBound97436
def owner : Owner := ⟨.program ⟨214⟩, ⟨11742⟩⟩
def transferEvent : Nat := 97436
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 97434 .coefficient, .predecessor 1 97435 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97434 .coefficient)
      LeftBound97432.bound (LeftBound97432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97432.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97432.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97435 .coefficient)
      LeftBound9970.bound (LeftBound9970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9971RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9970.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97432.bound, LeftBound9970.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97432.bound, LeftBound9970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97432.actual selector witness, LeftBound9970.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97436

namespace LeftBound97437
def owner : Owner := ⟨.program ⟨214⟩, ⟨11742⟩⟩
def transferEvent : Nat := 97437
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨97⟩⟩]⟩ [⟨.result 9971 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9971 .coefficient)
      LeftBound9970.bound (LeftBound9970.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨97⟩⟩) (rawTerms := some (Proof.Events038.exact9971RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9970.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound9970.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound9970.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound97437

namespace LeftBound97442
def owner : Owner := ⟨.program ⟨214⟩, ⟨11743⟩⟩
def transferEvent : Nat := 97442
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 97440 .coefficient) (.predecessor 1 97441 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97440 .coefficient)
      LeftBound97436.bound (LeftBound97436.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97439RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97436.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97436.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97441 .coefficient)
      LeftAuthority4729.bound (LeftAuthority4729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4730RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4729.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4729.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound97436.bound LeftAuthority4729.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97436.bound, LeftAuthority4729.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound97436.actual selector witness) * (LeftAuthority4729.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97442

namespace LeftBound97443
def owner : Owner := ⟨.program ⟨214⟩, ⟨11743⟩⟩
def transferEvent : Nat := 97443
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9595⟩⟩], []⟩ [⟨.result 4730 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4730 .coefficient)
      LeftAuthority4729.bound (LeftAuthority4729.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9595⟩⟩) (rawTerms := some (Proof.Events018.exact4730RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4729.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4729.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4729.bound []
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4729.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority4729.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound97443

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
