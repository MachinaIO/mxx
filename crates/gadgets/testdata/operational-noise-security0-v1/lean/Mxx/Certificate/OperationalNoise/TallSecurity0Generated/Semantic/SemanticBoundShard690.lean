import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard689

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound100282
def owner : Owner := ⟨.program ⟨214⟩, ⟨20959⟩⟩
def transferEvent : Nat := 100282
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 100280 .coefficient) (.value (.predecessor 1 100281 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100280 .coefficient)
      LeftAuthority100278.bound (LeftAuthority100278.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100279RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100278.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100278.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100281 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority100278.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100278.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority100278.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound100282

namespace LeftBound100286
def owner : Owner := ⟨.program ⟨214⟩, ⟨20960⟩⟩
def transferEvent : Nat := 100286
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 100284 .coefficient) (.predecessor 1 100285 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100284 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100285 .coefficient)
      LeftBound100282.bound (LeftBound100282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100282.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100282.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound100282.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound100282.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound100282.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100286

namespace LeftBound100287
def owner : Owner := ⟨.program ⟨214⟩, ⟨20960⟩⟩
def transferEvent : Nat := 100287
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20957⟩⟩]⟩ [⟨.result 100279 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100279 .coefficient)
      LeftAuthority100278.bound (LeftAuthority100278.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20957⟩⟩) (rawTerms := some (Proof.Events391.exact100279RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100278.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100278.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority100278.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100278.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority100278.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound100287

namespace LeftBound100288
def owner : Owner := ⟨.program ⟨214⟩, ⟨20960⟩⟩
def transferEvent : Nat := 100288
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 100287) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 100287)
      LeftBound100287.bound (LeftBound100287.actual selector witness) := by
  exact .transfer (LeftBound100287.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound100287.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound100287.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound100287.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100288

namespace LeftBound100359
def owner : Owner := ⟨.program ⟨214⟩, ⟨15574⟩⟩
def transferEvent : Nat := 100359
def frameStart : Nat := 100332
def rule : BoundRule := .identity (.predecessor 0 100358 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100358 .coefficient)
      LeftAuthority100356.bound (LeftAuthority100356.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100357RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100356.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100356.derived selector witness)

def rawBound : CoeffClass := LeftAuthority100356.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100356.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority100356.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound100359

namespace LeftBound100376
def owner : Owner := ⟨.program ⟨214⟩, ⟨15650⟩⟩
def transferEvent : Nat := 100376
def frameStart : Nat := 100332
def rule : BoundRule := .sum [.predecessor 0 100374 .coefficient, .predecessor 1 100375 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100374 .coefficient)
      LeftBound100359.bound (LeftBound100359.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound100359.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100375 .coefficient)
      LeftAuthority100372.bound (LeftAuthority100372.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority100372.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100359.bound, LeftAuthority100372.bound]
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100359.bound, LeftAuthority100372.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100359.actual selector witness, LeftAuthority100372.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100376

namespace LeftBound100379
def owner : Owner := ⟨.program ⟨214⟩, ⟨15651⟩⟩
def transferEvent : Nat := 100379
def frameStart : Nat := 100332
def rule : BoundRule := .identity (.predecessor 0 100378 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100378 .coefficient)
      LeftBound100376.bound (LeftBound100376.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound100376.derived selector witness)

def rawBound : CoeffClass := LeftBound100376.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100376.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound100376.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound100379

namespace LeftBound100385
def owner : Owner := ⟨.program ⟨214⟩, ⟨15652⟩⟩
def transferEvent : Nat := 100385
def frameStart : Nat := 100332
def rule : BoundRule := .product (.predecessor 0 100383 .coefficient) (.predecessor 1 100384 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100383 .coefficient)
      LeftAuthority100381.bound (LeftAuthority100381.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100382RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100381.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100381.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100384 .coefficient)
      LeftBound100379.bound (LeftBound100379.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100380RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100379.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100379.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority100381.bound LeftBound100379.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100381.bound, LeftBound100379.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority100381.actual selector witness) * (LeftBound100379.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100385

namespace LeftBound100393
def owner : Owner := ⟨.program ⟨214⟩, ⟨15653⟩⟩
def transferEvent : Nat := 100393
def frameStart : Nat := 100332
def rule : BoundRule := .sum [.predecessor 0 100391 .coefficient, .predecessor 1 100392 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100391 .coefficient)
      LeftAuthority100389.bound (LeftAuthority100389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100390RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100389.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100389.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100392 .coefficient)
      LeftBound100385.bound (LeftBound100385.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100385.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100385.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority100389.bound, LeftBound100385.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100389.bound, LeftBound100385.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority100389.actual selector witness, LeftBound100385.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100393

namespace LeftBound100397
def owner : Owner := ⟨.program ⟨214⟩, ⟨27181⟩⟩
def transferEvent : Nat := 100397
def frameStart : Nat := 100332
def rule : BoundRule := .product (.predecessor 0 100395 .coefficient) (.predecessor 1 100396 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100395 .coefficient)
      LeftBound100393.bound (LeftBound100393.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100394RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100393.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100393.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100396 .coefficient)
      LeftAuthority100370.bound (LeftAuthority100370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100371RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100370.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100370.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound100393.bound LeftAuthority100370.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100393.bound, LeftAuthority100370.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound100393.actual selector witness) * (LeftAuthority100370.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100397

namespace LeftBound100408
def owner : Owner := ⟨.program ⟨214⟩, ⟨15623⟩⟩
def transferEvent : Nat := 100408
def frameStart : Nat := 100332
def rule : BoundRule := .product (.predecessor 0 100406 .coefficient) (.predecessor 1 100407 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100406 .coefficient)
      LeftAuthority100381.bound (LeftAuthority100381.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100382RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100381.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100381.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100407 .coefficient)
      LeftAuthority100404.bound (LeftAuthority100404.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100405RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100404.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100404.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority100381.bound LeftAuthority100404.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100381.bound, LeftAuthority100404.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority100381.actual selector witness) * (LeftAuthority100404.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100408

namespace LeftBound100416
def owner : Owner := ⟨.program ⟨214⟩, ⟨15624⟩⟩
def transferEvent : Nat := 100416
def frameStart : Nat := 100332
def rule : BoundRule := .sum [.predecessor 0 100414 .coefficient, .predecessor 1 100415 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100414 .coefficient)
      LeftAuthority100412.bound (LeftAuthority100412.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100413RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100412.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100412.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100415 .coefficient)
      LeftBound100408.bound (LeftBound100408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100408.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100408.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority100412.bound, LeftBound100408.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100412.bound, LeftBound100408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority100412.actual selector witness, LeftBound100408.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100416

namespace LeftBound100420
def owner : Owner := ⟨.program ⟨214⟩, ⟨27185⟩⟩
def transferEvent : Nat := 100420
def frameStart : Nat := 100332
def rule : BoundRule := .sum [.predecessor 0 100418 .coefficient, .predecessor 1 100419 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100418 .coefficient)
      LeftBound100416.bound (LeftBound100416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100417RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100416.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100416.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100419 .coefficient)
      LeftBound100397.bound (LeftBound100397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100397.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100416.bound, LeftBound100397.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100416.bound, LeftBound100397.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100416.actual selector witness, LeftBound100397.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100420

namespace LeftBound100433
def owner : Owner := ⟨.program ⟨214⟩, ⟨27183⟩⟩
def transferEvent : Nat := 100433
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 100431 .coefficient, .predecessor 1 100432 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100431 .coefficient)
      LeftBound100286.bound (LeftBound100286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100430RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100286.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100286.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100432 .coefficient)
      LeftBound100269.bound (LeftBound100269.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100276RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100269.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100269.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100286.bound, LeftBound100269.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100286.bound, LeftBound100269.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100286.actual selector witness, LeftBound100269.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100433

namespace LeftBound100436
def owner : Owner := ⟨.program ⟨214⟩, ⟨27183⟩⟩
def transferEvent : Nat := 100436
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 100430 .summary, .result 100276 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100430 .summary)
      LeftBound100288.bound (LeftBound100288.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20960⟩⟩) (rawTerms := some (Proof.Events392.exact100430RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100288.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100276 .summary)
      LeftBound100271.bound (LeftBound100271.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27182⟩⟩) (rawTerms := some (Proof.Events391.exact100276RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100271.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100288.bound, LeftBound100271.bound]
def bound : CoeffClass := .finite ⟨1291978824159503986688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100288.bound, LeftBound100271.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100288.actual selector witness, LeftBound100271.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100436

namespace LeftBound100460
def owner : Owner := ⟨.program ⟨214⟩, ⟨11122⟩⟩
def transferEvent : Nat := 100460
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 100458 .coefficient) (.predecessor 1 100459 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100458 .coefficient)
      LeftAuthority4887.bound (LeftAuthority4887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact4888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4887.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4887.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100459 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4887.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4887.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4887.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound100460

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
