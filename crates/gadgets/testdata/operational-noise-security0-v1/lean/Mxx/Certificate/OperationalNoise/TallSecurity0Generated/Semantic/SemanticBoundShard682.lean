import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard681

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound99345
def owner : Owner := ⟨.program ⟨214⟩, ⟨6758⟩⟩
def transferEvent : Nat := 99345
def frameStart : Nat := 99279
def rule : BoundRule := .identity (.predecessor 0 99344 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99344 .coefficient)
      LeftAuthority99332.bound (LeftAuthority99332.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99333RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99332.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99332.derived selector witness)

def rawBound : CoeffClass := LeftAuthority99332.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99332.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority99332.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound99345

namespace LeftBound99349
def owner : Owner := ⟨.program ⟨214⟩, ⟨7851⟩⟩
def transferEvent : Nat := 99349
def frameStart : Nat := 99279
def rule : BoundRule := .product (.predecessor 0 99347 .coefficient) (.predecessor 1 99348 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99347 .coefficient)
      LeftBound99345.bound (LeftBound99345.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99346RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99345.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99345.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99348 .coefficient)
      LeftBound99342.bound (LeftBound99342.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99342.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99342.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99345.bound LeftBound99342.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99345.bound, LeftBound99342.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99345.actual selector witness) * (LeftBound99342.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99349

namespace LeftBound99354
def owner : Owner := ⟨.program ⟨214⟩, ⟨14092⟩⟩
def transferEvent : Nat := 99354
def frameStart : Nat := 99279
def rule : BoundRule := .sum [.predecessor 0 99352 .coefficient, .predecessor 1 99353 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99352 .coefficient)
      LeftBound99349.bound (LeftBound99349.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99351RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99349.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99349.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99353 .coefficient)
      LeftBound99326.bound (LeftBound99326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99328RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99326.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99326.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99349.bound, LeftBound99326.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99349.bound, LeftBound99326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99349.actual selector witness, LeftBound99326.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99354

namespace LeftBound99358
def owner : Owner := ⟨.program ⟨214⟩, ⟨25979⟩⟩
def transferEvent : Nat := 99358
def frameStart : Nat := 99279
def rule : BoundRule := .product (.predecessor 0 99356 .coefficient) (.predecessor 1 99357 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99356 .coefficient)
      LeftBound99354.bound (LeftBound99354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99355RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99354.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99357 .coefficient)
      LeftAuthority99311.bound (LeftAuthority99311.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99312RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99311.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99311.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99354.bound LeftAuthority99311.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99354.bound, LeftAuthority99311.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99354.actual selector witness) * (LeftAuthority99311.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99358

namespace LeftBound99369
def owner : Owner := ⟨.program ⟨214⟩, ⟨15813⟩⟩
def transferEvent : Nat := 99369
def frameStart : Nat := 99279
def rule : BoundRule := .product (.predecessor 0 99367 .coefficient) (.predecessor 1 99368 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99367 .coefficient)
      LeftAuthority99322.bound (LeftAuthority99322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99322.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99322.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99368 .coefficient)
      LeftAuthority99365.bound (LeftAuthority99365.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99366RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99365.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99365.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority99322.bound LeftAuthority99365.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99322.bound, LeftAuthority99365.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority99322.actual selector witness) * (LeftAuthority99365.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99369

namespace LeftBound99377
def owner : Owner := ⟨.program ⟨214⟩, ⟨15814⟩⟩
def transferEvent : Nat := 99377
def frameStart : Nat := 99279
def rule : BoundRule := .sum [.predecessor 0 99375 .coefficient, .predecessor 1 99376 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99375 .coefficient)
      LeftAuthority99373.bound (LeftAuthority99373.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99374RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99373.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99373.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99376 .coefficient)
      LeftBound99369.bound (LeftBound99369.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99371RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99369.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99369.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority99373.bound, LeftBound99369.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99373.bound, LeftBound99369.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority99373.actual selector witness, LeftBound99369.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99377

namespace LeftBound99381
def owner : Owner := ⟨.program ⟨214⟩, ⟨25980⟩⟩
def transferEvent : Nat := 99381
def frameStart : Nat := 99279
def rule : BoundRule := .sum [.predecessor 0 99379 .coefficient, .predecessor 1 99380 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99379 .coefficient)
      LeftBound99377.bound (LeftBound99377.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99378RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99377.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99377.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99380 .coefficient)
      LeftBound99358.bound (LeftBound99358.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99363RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99358.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99358.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99377.bound, LeftBound99358.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99377.bound, LeftBound99358.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99377.actual selector witness, LeftBound99358.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99381

namespace LeftBound99394
def owner : Owner := ⟨.program ⟨214⟩, ⟨25978⟩⟩
def transferEvent : Nat := 99394
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99392 .coefficient, .predecessor 1 99393 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99392 .coefficient)
      LeftBound99239.bound (LeftBound99239.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99391RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99239.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99239.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99393 .coefficient)
      LeftBound99222.bound (LeftBound99222.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99229RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99222.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99222.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99239.bound, LeftBound99222.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99239.bound, LeftBound99222.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99239.actual selector witness, LeftBound99222.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99394

namespace LeftBound99397
def owner : Owner := ⟨.program ⟨214⟩, ⟨25978⟩⟩
def transferEvent : Nat := 99397
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99391 .summary, .result 99229 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99391 .summary)
      LeftBound99241.bound (LeftBound99241.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19448⟩⟩) (rawTerms := some (Proof.Events388.exact99391RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99241.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99229 .summary)
      LeftBound99224.bound (LeftBound99224.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25977⟩⟩) (rawTerms := some (Proof.Events387.exact99229RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99224.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99241.bound, LeftBound99224.bound]
def bound : CoeffClass := .finite ⟨352054612209664, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99241.bound, LeftBound99224.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99241.actual selector witness, LeftBound99224.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99397

namespace LeftBound99401
def owner : Owner := ⟨.program ⟨214⟩, ⟨27616⟩⟩
def transferEvent : Nat := 99401
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 99399 .coefficient) (.predecessor 1 99400 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99399 .coefficient)
      LeftBound99394.bound (LeftBound99394.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99394.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99394.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99400 .coefficient)
      LeftAuthority99144.bound (LeftAuthority99144.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events387.exact99145RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99144.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99144.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99394.bound LeftAuthority99144.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99394.bound, LeftAuthority99144.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99394.actual selector witness) * (LeftAuthority99144.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99401

namespace LeftBound99402
def owner : Owner := ⟨.program ⟨214⟩, ⟨27616⟩⟩
def transferEvent : Nat := 99402
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27614⟩⟩]⟩ [⟨.result 99145 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99145 .coefficient)
      LeftAuthority99144.bound (LeftAuthority99144.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27614⟩⟩) (rawTerms := some (Proof.Events387.exact99145RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99144.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99144.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority99144.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99144.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority99144.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound99402

namespace LeftBound99403
def owner : Owner := ⟨.program ⟨214⟩, ⟨27616⟩⟩
def transferEvent : Nat := 99403
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 99398 .summary) (.transfer 99402) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99398 .summary)
      LeftBound99397.bound (LeftBound99397.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25978⟩⟩) (rawTerms := some (Proof.Events388.exact99398RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 99402)
      LeftBound99402.bound (LeftBound99402.actual selector witness) := by
  exact .transfer (LeftBound99402.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99397.bound LeftBound99402.bound
def bound : CoeffClass := .finite ⟨1292046059683262234624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99397.bound, LeftBound99402.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99397.actual selector witness) * (LeftBound99402.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99403

namespace LeftBound99414
def owner : Owner := ⟨.program ⟨214⟩, ⟨21247⟩⟩
def transferEvent : Nat := 99414
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 99412 .coefficient) (.value (.predecessor 1 99413 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99412 .coefficient)
      LeftAuthority99410.bound (LeftAuthority99410.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99411RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99410.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99410.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99413 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority99410.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99410.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority99410.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound99414

namespace LeftBound99418
def owner : Owner := ⟨.program ⟨214⟩, ⟨21248⟩⟩
def transferEvent : Nat := 99418
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 99416 .coefficient) (.predecessor 1 99417 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99416 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99417 .coefficient)
      LeftBound99414.bound (LeftBound99414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99414.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99414.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound99414.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound99414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound99414.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99418

namespace LeftBound99419
def owner : Owner := ⟨.program ⟨214⟩, ⟨21248⟩⟩
def transferEvent : Nat := 99419
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21245⟩⟩]⟩ [⟨.result 99411 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99411 .coefficient)
      LeftAuthority99410.bound (LeftAuthority99410.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21245⟩⟩) (rawTerms := some (Proof.Events388.exact99411RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99410.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99410.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority99410.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority99410.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound99419

namespace LeftBound99420
def owner : Owner := ⟨.program ⟨214⟩, ⟨21248⟩⟩
def transferEvent : Nat := 99420
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 99419) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 99419)
      LeftBound99419.bound (LeftBound99419.actual selector witness) := by
  exact .transfer (LeftBound99419.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound99419.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound99419.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound99419.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99420

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
