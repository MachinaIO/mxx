import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard012
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard013

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound4493
def owner : Owner := ⟨.program ⟨214⟩, ⟨17819⟩⟩
def transferEvent : Nat := 4493
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4491 .coefficient, .predecessor 1 4492 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4491 .coefficient)
      LeftBound4489.bound (LeftBound4489.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4489.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4489.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4492 .coefficient)
      LeftBound4404.bound (LeftBound4404.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4404.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4404.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4489.bound, LeftBound4404.bound]
def bound : CoeffClass := .finite ⟨1593837033067242249035979, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4489.bound, LeftBound4404.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound4489.actual selector witness, LeftBound4404.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4493

namespace LeftBound4497
def owner : Owner := ⟨.program ⟨214⟩, ⟨18037⟩⟩
def transferEvent : Nat := 4497
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4495 .coefficient, .predecessor 1 4496 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4495 .coefficient)
      LeftBound4493.bound (LeftBound4493.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4493.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4496 .coefficient)
      LeftBound4396.bound (LeftBound4396.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4396.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4396.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4493.bound, LeftBound4396.bound]
def bound : CoeffClass := .finite ⟨1818214806102629497873539, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4493.bound, LeftBound4396.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound4493.actual selector witness, LeftBound4396.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4497

namespace LeftBound4501
def owner : Owner := ⟨.program ⟨214⟩, ⟨18038⟩⟩
def transferEvent : Nat := 4501
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4499 .coefficient, .predecessor 1 4500 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4499 .coefficient)
      LeftBound4497.bound (LeftBound4497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4497.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4497.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4500 .coefficient)
      LeftBound4388.bound (LeftBound4388.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4390RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4388.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4388.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4497.bound, LeftBound4388.bound]
def bound : CoeffClass := .finite ⟨2044702714934587786668819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4497.bound, LeftBound4388.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound4497.actual selector witness, LeftBound4388.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4501

namespace LeftBound4505
def owner : Owner := ⟨.program ⟨214⟩, ⟨18039⟩⟩
def transferEvent : Nat := 4505
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4503 .coefficient, .predecessor 1 4504 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4503 .coefficient)
      LeftBound4501.bound (LeftBound4501.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4502RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4501.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4501.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4504 .coefficient)
      LeftBound4380.bound (LeftBound4380.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4382RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4380.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4380.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4501.bound, LeftBound4380.bound]
def bound : CoeffClass := .finite ⟨2271712485307633536959019, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4501.bound, LeftBound4380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound4501.actual selector witness, LeftBound4380.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4505

namespace LeftBound4509
def owner : Owner := ⟨.program ⟨214⟩, ⟨18834⟩⟩
def transferEvent : Nat := 4509
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4507 .coefficient, .predecessor 1 4508 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4507 .coefficient)
      LeftBound4505.bound (LeftBound4505.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4506RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4505.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4505.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4508 .coefficient)
      LeftBound4372.bound (LeftBound4372.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4374RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4372.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4372.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4505.bound, LeftBound4372.bound]
def bound : CoeffClass := .finite ⟨2499949335520533588602139, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4505.bound, LeftBound4372.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound4505.actual selector witness, LeftBound4372.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4509

namespace LeftBound4513
def owner : Owner := ⟨.program ⟨214⟩, ⟨18835⟩⟩
def transferEvent : Nat := 4513
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4511 .coefficient, .predecessor 1 4512 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4511 .coefficient)
      LeftBound4509.bound (LeftBound4509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4512 .coefficient)
      LeftBound4364.bound (LeftBound4364.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4366RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4364.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4364.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4509.bound, LeftBound4364.bound]
def bound : CoeffClass := .finite ⟨2728804713782791092959739, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4509.bound, LeftBound4364.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound4509.actual selector witness, LeftBound4364.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4513

namespace LeftBound4517
def owner : Owner := ⟨.program ⟨214⟩, ⟨18836⟩⟩
def transferEvent : Nat := 4517
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4515 .coefficient, .predecessor 1 4516 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4515 .coefficient)
      LeftBound4513.bound (LeftBound4513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4514RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4513.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4513.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4516 .coefficient)
      LeftBound4356.bound (LeftBound4356.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4356.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4356.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4513.bound, LeftBound4356.bound]
def bound : CoeffClass := .finite ⟨2957926202950004710694499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4513.bound, LeftBound4356.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound4513.actual selector witness, LeftBound4356.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4517

namespace LeftBound4521
def owner : Owner := ⟨.program ⟨214⟩, ⟨18837⟩⟩
def transferEvent : Nat := 4521
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4519 .coefficient, .predecessor 1 4520 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4519 .coefficient)
      LeftBound4517.bound (LeftBound4517.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4517.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4517.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4520 .coefficient)
      LeftBound4348.bound (LeftBound4348.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4350RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4348.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4348.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4517.bound, LeftBound4348.bound]
def bound : CoeffClass := .finite ⟨3187511970717354526236219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4517.bound, LeftBound4348.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound4517.actual selector witness, LeftBound4348.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4521

namespace LeftBound4525
def owner : Owner := ⟨.program ⟨214⟩, ⟨18838⟩⟩
def transferEvent : Nat := 4525
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4523 .coefficient, .predecessor 1 4524 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4523 .coefficient)
      LeftBound4521.bound (LeftBound4521.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4522RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4521.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4521.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4524 .coefficient)
      LeftBound4340.bound (LeftBound4340.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4342RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4340.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4340.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4521.bound, LeftBound4340.bound]
def bound : CoeffClass := .finite ⟨3417662756781096507033579, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4521.bound, LeftBound4340.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound4521.actual selector witness, LeftBound4340.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4525

namespace LeftBound4529
def owner : Owner := ⟨.program ⟨214⟩, ⟨18839⟩⟩
def transferEvent : Nat := 4529
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4527 .coefficient, .predecessor 1 4528 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4527 .coefficient)
      LeftBound4525.bound (LeftBound4525.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4526RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4525.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4525.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4528 .coefficient)
      LeftBound4332.bound (LeftBound4332.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4334RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4332.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4332.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4525.bound, LeftBound4332.bound]
def bound : CoeffClass := .finite ⟨3648263642165693263543059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4525.bound, LeftBound4332.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound4525.actual selector witness, LeftBound4332.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4529

namespace LeftBound4533
def owner : Owner := ⟨.program ⟨214⟩, ⟨18840⟩⟩
def transferEvent : Nat := 4533
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4531 .coefficient, .predecessor 1 4532 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4531 .coefficient)
      LeftBound4529.bound (LeftBound4529.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4529.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4529.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4532 .coefficient)
      LeftBound4324.bound (LeftBound4324.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4326RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4324.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4324.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4529.bound, LeftBound4324.bound]
def bound : CoeffClass := .finite ⟨3878994884184198780231459, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4529.bound, LeftBound4324.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound4529.actual selector witness, LeftBound4324.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4533

namespace LeftBound4537
def owner : Owner := ⟨.program ⟨214⟩, ⟨18842⟩⟩
def transferEvent : Nat := 4537
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4535 .coefficient, .predecessor 1 4536 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4535 .coefficient)
      LeftBound4533.bound (LeftBound4533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4534RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4533.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4533.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4536 .coefficient)
      LeftBound4316.bound (LeftBound4316.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4318RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4316.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4316.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4533.bound, LeftBound4316.bound]
def bound : CoeffClass := .finite ⟨8101376613122849735629179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4533.bound, LeftBound4316.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound4533.actual selector witness, LeftBound4316.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4537

namespace LeftBound4541
def owner : Owner := ⟨.program ⟨214⟩, ⟨18843⟩⟩
def transferEvent : Nat := 4541
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 4539 .coefficient) (.predecessor 1 4540 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4539 .coefficient)
      LeftBound4537.bound (LeftBound4537.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4537.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4537.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4540 .coefficient)
      LeftAuthority3820.bound (LeftAuthority3820.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events014.exact3821RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3820.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3820.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound4537.bound LeftAuthority3820.bound
def bound : CoeffClass := .finite ⟨4121992727563839716010138668593533682710543272274454902509901787881037087030242311712, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4537.bound, LeftAuthority3820.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound4537.actual selector witness) * (LeftAuthority3820.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound4541

namespace LeftBound5054
def owner : Owner := ⟨.program ⟨214⟩, ⟨18486⟩⟩
def transferEvent : Nat := 5054
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 5052 .coefficient) (.predecessor 1 5053 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5052 .coefficient)
      LeftAuthority5050.bound (LeftAuthority5050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact5051RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5050.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5050.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5053 .coefficient)
      LeftAuthority35.bound (LeftAuthority35.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact36RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority5050.bound LeftAuthority35.bound
def bound : CoeffClass := .finite ⟨4222381728938650955397720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5050.bound, LeftAuthority35.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority5050.actual selector witness) * (LeftAuthority35.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound5054

namespace LeftBound5062
def owner : Owner := ⟨.program ⟨214⟩, ⟨18115⟩⟩
def transferEvent : Nat := 5062
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 5060 .coefficient) (.predecessor 1 5061 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5060 .coefficient)
      LeftAuthority5058.bound (LeftAuthority5058.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact5059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5058.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5058.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5061 .coefficient)
      LeftAuthority542.bound (LeftAuthority542.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority542.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority542.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority5058.bound LeftAuthority542.bound
def bound : CoeffClass := .finite ⟨230731242018505516688400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5058.bound, LeftAuthority542.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority5058.actual selector witness) * (LeftAuthority542.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound5062

namespace LeftBound5070
def owner : Owner := ⟨.program ⟨214⟩, ⟨16918⟩⟩
def transferEvent : Nat := 5070
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 5068 .coefficient) (.predecessor 1 5069 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5068 .coefficient)
      LeftAuthority5066.bound (LeftAuthority5066.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact5067RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5066.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5066.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5069 .coefficient)
      LeftAuthority552.bound (LeftAuthority552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority552.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority552.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority5066.bound LeftAuthority552.bound
def bound : CoeffClass := .finite ⟨230600885384596756509480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5066.bound, LeftAuthority552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority5066.actual selector witness) * (LeftAuthority552.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound5070

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
