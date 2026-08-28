import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard087
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard088
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard493

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound72455
def owner : Owner := ⟨.program ⟨214⟩, ⟨15458⟩⟩
def transferEvent : Nat := 72455
def frameStart : Nat := 72399
def rule : BoundRule := .sum [.predecessor 0 72453 .coefficient, .predecessor 1 72454 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72453 .coefficient)
      LeftBound72438.bound (LeftBound72438.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound72438.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72454 .coefficient)
      LeftAuthority72451.bound (LeftAuthority72451.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority72451.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72438.bound, LeftAuthority72451.bound]
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72438.bound, LeftAuthority72451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72438.actual selector witness, LeftAuthority72451.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72455

namespace LeftBound72458
def owner : Owner := ⟨.program ⟨214⟩, ⟨15459⟩⟩
def transferEvent : Nat := 72458
def frameStart : Nat := 72399
def rule : BoundRule := .identity (.predecessor 0 72457 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72457 .coefficient)
      LeftBound72455.bound (LeftBound72455.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound72455.derived selector witness)

def rawBound : CoeffClass := LeftBound72455.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72455.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound72455.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound72458

namespace LeftBound72464
def owner : Owner := ⟨.program ⟨214⟩, ⟨15460⟩⟩
def transferEvent : Nat := 72464
def frameStart : Nat := 72399
def rule : BoundRule := .product (.predecessor 0 72462 .coefficient) (.predecessor 1 72463 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72462 .coefficient)
      LeftAuthority72460.bound (LeftAuthority72460.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72461RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72460.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72463 .coefficient)
      LeftBound72458.bound (LeftBound72458.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72459RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72458.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72458.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority72460.bound LeftBound72458.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72460.bound, LeftBound72458.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority72460.actual selector witness) * (LeftBound72458.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72464

namespace LeftBound72472
def owner : Owner := ⟨.program ⟨214⟩, ⟨15461⟩⟩
def transferEvent : Nat := 72472
def frameStart : Nat := 72399
def rule : BoundRule := .sum [.predecessor 0 72470 .coefficient, .predecessor 1 72471 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72470 .coefficient)
      LeftAuthority72468.bound (LeftAuthority72468.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72469RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72468.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72468.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72471 .coefficient)
      LeftBound72464.bound (LeftBound72464.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72466RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72464.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72464.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority72468.bound, LeftBound72464.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72468.bound, LeftBound72464.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority72468.actual selector witness, LeftBound72464.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72472

namespace LeftBound72476
def owner : Owner := ⟨.program ⟨214⟩, ⟨26986⟩⟩
def transferEvent : Nat := 72476
def frameStart : Nat := 72399
def rule : BoundRule := .product (.predecessor 0 72474 .coefficient) (.predecessor 1 72475 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72474 .coefficient)
      LeftBound72472.bound (LeftBound72472.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72473RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72472.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72472.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72475 .coefficient)
      LeftAuthority72449.bound (LeftAuthority72449.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72450RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72449.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72449.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound72472.bound LeftAuthority72449.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72472.bound, LeftAuthority72449.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound72472.actual selector witness) * (LeftAuthority72449.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72476

namespace LeftBound72487
def owner : Owner := ⟨.program ⟨214⟩, ⟨17325⟩⟩
def transferEvent : Nat := 72487
def frameStart : Nat := 72399
def rule : BoundRule := .product (.predecessor 0 72485 .coefficient) (.predecessor 1 72486 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72485 .coefficient)
      LeftAuthority72460.bound (LeftAuthority72460.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72461RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72460.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72486 .coefficient)
      LeftAuthority72483.bound (LeftAuthority72483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72483.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72483.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority72460.bound LeftAuthority72483.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72460.bound, LeftAuthority72483.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority72460.actual selector witness) * (LeftAuthority72483.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72487

namespace LeftBound72495
def owner : Owner := ⟨.program ⟨214⟩, ⟨17326⟩⟩
def transferEvent : Nat := 72495
def frameStart : Nat := 72399
def rule : BoundRule := .sum [.predecessor 0 72493 .coefficient, .predecessor 1 72494 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72493 .coefficient)
      LeftAuthority72491.bound (LeftAuthority72491.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72492RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72491.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72491.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72494 .coefficient)
      LeftBound72487.bound (LeftBound72487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72489RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72487.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72487.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority72491.bound, LeftBound72487.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72491.bound, LeftBound72487.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority72491.actual selector witness, LeftBound72487.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72495

namespace LeftBound72499
def owner : Owner := ⟨.program ⟨214⟩, ⟨26990⟩⟩
def transferEvent : Nat := 72499
def frameStart : Nat := 72399
def rule : BoundRule := .sum [.predecessor 0 72497 .coefficient, .predecessor 1 72498 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72497 .coefficient)
      LeftBound72495.bound (LeftBound72495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72496RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72495.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72495.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72498 .coefficient)
      LeftBound72476.bound (LeftBound72476.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72481RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72476.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72476.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72495.bound, LeftBound72476.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72495.bound, LeftBound72476.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72495.actual selector witness, LeftBound72476.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72499

namespace LeftBound72512
def owner : Owner := ⟨.program ⟨214⟩, ⟨26988⟩⟩
def transferEvent : Nat := 72512
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 72510 .coefficient, .predecessor 1 72511 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72510 .coefficient)
      LeftBound72341.bound (LeftBound72341.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72509RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72341.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72341.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72511 .coefficient)
      LeftBound72324.bound (LeftBound72324.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events282.exact72331RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72324.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72324.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72341.bound, LeftBound72324.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72341.bound, LeftBound72324.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72341.actual selector witness, LeftBound72324.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72512

namespace LeftBound72515
def owner : Owner := ⟨.program ⟨214⟩, ⟨26988⟩⟩
def transferEvent : Nat := 72515
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 72509 .summary, .result 72331 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72509 .summary)
      LeftBound72343.bound (LeftBound72343.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20823⟩⟩) (rawTerms := some (Proof.Events283.exact72509RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72343.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72331 .summary)
      LeftBound72326.bound (LeftBound72326.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26987⟩⟩) (rawTerms := some (Proof.Events282.exact72331RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72326.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72343.bound, LeftBound72326.bound]
def bound : CoeffClass := .finite ⟨1291933999269462814720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72343.bound, LeftBound72326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72343.actual selector witness, LeftBound72326.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72515

namespace LeftBound72539
def owner : Owner := ⟨.program ⟨214⟩, ⟨10972⟩⟩
def transferEvent : Nat := 72539
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 72537 .coefficient) (.predecessor 1 72538 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72537 .coefficient)
      LeftAuthority3430.bound (LeftAuthority3430.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3431RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3430.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3430.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72538 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3430.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3430.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3430.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound72539

namespace LeftBound72544
def owner : Owner := ⟨.program ⟨214⟩, ⟨7192⟩⟩
def transferEvent : Nat := 72544
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 72542 .coefficient) (.predecessor 1 72543 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72542 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72543 .coefficient)
      LeftBound13986.bound (LeftBound13986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13987RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13986.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13986.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound13986.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound13986.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound13986.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72544

namespace LeftBound72549
def owner : Owner := ⟨.program ⟨214⟩, ⟨10973⟩⟩
def transferEvent : Nat := 72549
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 72547 .coefficient, .predecessor 1 72548 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72547 .coefficient)
      LeftBound72544.bound (LeftBound72544.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72546RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72544.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72544.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72548 .coefficient)
      LeftBound72539.bound (LeftBound72539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72541RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72539.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72544.bound, LeftBound72539.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72544.bound, LeftBound72539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72544.actual selector witness, LeftBound72539.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72549

namespace LeftBound72553
def owner : Owner := ⟨.program ⟨214⟩, ⟨10974⟩⟩
def transferEvent : Nat := 72553
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 72551 .coefficient, .predecessor 1 72552 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72551 .coefficient)
      LeftBound72549.bound (LeftBound72549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72549.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72549.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72552 .coefficient)
      LeftBound13978.bound (LeftBound13978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13978.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72549.bound, LeftBound13978.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72549.bound, LeftBound13978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72549.actual selector witness, LeftBound13978.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72553

namespace LeftBound72554
def owner : Owner := ⟨.program ⟨214⟩, ⟨10974⟩⟩
def transferEvent : Nat := 72554
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨88⟩⟩]⟩ [⟨.result 13979 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13979 .coefficient)
      LeftBound13978.bound (LeftBound13978.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨88⟩⟩) (rawTerms := some (Proof.Events054.exact13979RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13978.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound13978.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound13978.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound72554

namespace LeftBound72559
def owner : Owner := ⟨.program ⟨214⟩, ⟨10975⟩⟩
def transferEvent : Nat := 72559
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 72557 .coefficient) (.predecessor 1 72558 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72557 .coefficient)
      LeftBound72553.bound (LeftBound72553.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72556RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72553.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72553.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72558 .coefficient)
      LeftAuthority3433.bound (LeftAuthority3433.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3434RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3433.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3433.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound72553.bound LeftAuthority3433.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72553.bound, LeftAuthority3433.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound72553.actual selector witness) * (LeftAuthority3433.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72559

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
