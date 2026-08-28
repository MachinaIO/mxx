import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard475
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard521

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound77412
def owner : Owner := ⟨.program ⟨214⟩, ⟨16175⟩⟩
def transferEvent : Nat := 77412
def frameStart : Nat := 77373
def rule : BoundRule := .identity (.predecessor 0 77411 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77411 .coefficient)
      LeftAuthority77409.bound (LeftAuthority77409.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77409.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77409.derived selector witness)

def rawBound : CoeffClass := LeftAuthority77409.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77409.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority77409.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound77412

namespace LeftBound77429
def owner : Owner := ⟨.program ⟨214⟩, ⟨16214⟩⟩
def transferEvent : Nat := 77429
def frameStart : Nat := 77373
def rule : BoundRule := .sum [.predecessor 0 77427 .coefficient, .predecessor 1 77428 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77427 .coefficient)
      LeftBound77412.bound (LeftBound77412.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound77412.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77428 .coefficient)
      LeftAuthority77425.bound (LeftAuthority77425.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority77425.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77412.bound, LeftAuthority77425.bound]
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77412.bound, LeftAuthority77425.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound77412.actual selector witness, LeftAuthority77425.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77429

namespace LeftBound77432
def owner : Owner := ⟨.program ⟨214⟩, ⟨16215⟩⟩
def transferEvent : Nat := 77432
def frameStart : Nat := 77373
def rule : BoundRule := .identity (.predecessor 0 77431 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77431 .coefficient)
      LeftBound77429.bound (LeftBound77429.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound77429.derived selector witness)

def rawBound : CoeffClass := LeftBound77429.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77429.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound77429.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound77432

namespace LeftBound77438
def owner : Owner := ⟨.program ⟨214⟩, ⟨16216⟩⟩
def transferEvent : Nat := 77438
def frameStart : Nat := 77373
def rule : BoundRule := .product (.predecessor 0 77436 .coefficient) (.predecessor 1 77437 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77436 .coefficient)
      LeftAuthority77434.bound (LeftAuthority77434.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77435RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77434.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77434.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77437 .coefficient)
      LeftBound77432.bound (LeftBound77432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77432.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77432.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority77434.bound LeftBound77432.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77434.bound, LeftBound77432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority77434.actual selector witness) * (LeftBound77432.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77438

namespace LeftBound77446
def owner : Owner := ⟨.program ⟨214⟩, ⟨16217⟩⟩
def transferEvent : Nat := 77446
def frameStart : Nat := 77373
def rule : BoundRule := .sum [.predecessor 0 77444 .coefficient, .predecessor 1 77445 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77444 .coefficient)
      LeftAuthority77442.bound (LeftAuthority77442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77443RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77442.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77442.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77445 .coefficient)
      LeftBound77438.bound (LeftBound77438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77440RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77438.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77438.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority77442.bound, LeftBound77438.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77442.bound, LeftBound77438.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority77442.actual selector witness, LeftBound77438.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77446

namespace LeftBound77450
def owner : Owner := ⟨.program ⟨214⟩, ⟨28281⟩⟩
def transferEvent : Nat := 77450
def frameStart : Nat := 77373
def rule : BoundRule := .product (.predecessor 0 77448 .coefficient) (.predecessor 1 77449 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77448 .coefficient)
      LeftBound77446.bound (LeftBound77446.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77447RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77446.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77446.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77449 .coefficient)
      LeftAuthority77423.bound (LeftAuthority77423.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77424RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77423.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77423.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound77446.bound LeftAuthority77423.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77446.bound, LeftAuthority77423.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound77446.actual selector witness) * (LeftAuthority77423.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77450

namespace LeftBound77461
def owner : Owner := ⟨.program ⟨214⟩, ⟨17660⟩⟩
def transferEvent : Nat := 77461
def frameStart : Nat := 77373
def rule : BoundRule := .product (.predecessor 0 77459 .coefficient) (.predecessor 1 77460 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77459 .coefficient)
      LeftAuthority77434.bound (LeftAuthority77434.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77435RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77434.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77434.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77460 .coefficient)
      LeftAuthority77457.bound (LeftAuthority77457.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77458RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77457.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77457.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority77434.bound LeftAuthority77457.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77434.bound, LeftAuthority77457.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority77434.actual selector witness) * (LeftAuthority77457.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77461

namespace LeftBound77469
def owner : Owner := ⟨.program ⟨214⟩, ⟨17661⟩⟩
def transferEvent : Nat := 77469
def frameStart : Nat := 77373
def rule : BoundRule := .sum [.predecessor 0 77467 .coefficient, .predecessor 1 77468 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77467 .coefficient)
      LeftAuthority77465.bound (LeftAuthority77465.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77466RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77465.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77465.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77468 .coefficient)
      LeftBound77461.bound (LeftBound77461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77461.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77461.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority77465.bound, LeftBound77461.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77465.bound, LeftBound77461.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority77465.actual selector witness, LeftBound77461.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77469

namespace LeftBound77473
def owner : Owner := ⟨.program ⟨214⟩, ⟨28286⟩⟩
def transferEvent : Nat := 77473
def frameStart : Nat := 77373
def rule : BoundRule := .sum [.predecessor 0 77471 .coefficient, .predecessor 1 77472 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77471 .coefficient)
      LeftBound77469.bound (LeftBound77469.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77470RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77469.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77469.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77472 .coefficient)
      LeftBound77450.bound (LeftBound77450.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77455RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77450.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77450.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77469.bound, LeftBound77450.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77469.bound, LeftBound77450.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound77469.actual selector witness, LeftBound77450.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77473

namespace LeftBound77486
def owner : Owner := ⟨.program ⟨214⟩, ⟨28283⟩⟩
def transferEvent : Nat := 77486
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 77484 .coefficient, .predecessor 1 77485 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77484 .coefficient)
      LeftBound77315.bound (LeftBound77315.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77315.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77315.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77485 .coefficient)
      LeftBound77298.bound (LeftBound77298.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events301.exact77305RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77298.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77298.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77315.bound, LeftBound77298.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77315.bound, LeftBound77298.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound77315.actual selector witness, LeftBound77298.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77486

namespace LeftBound77489
def owner : Owner := ⟨.program ⟨214⟩, ⟨28283⟩⟩
def transferEvent : Nat := 77489
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 77483 .summary, .result 77305 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77483 .summary)
      LeftBound77317.bound (LeftBound77317.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21615⟩⟩) (rawTerms := some (Proof.Events302.exact77483RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77317.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77305 .summary)
      LeftBound77300.bound (LeftBound77300.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28282⟩⟩) (rawTerms := some (Proof.Events301.exact77305RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77300.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77317.bound, LeftBound77300.bound]
def bound : CoeffClass := .finite ⟨1292180536164689260544, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77317.bound, LeftBound77300.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound77317.actual selector witness, LeftBound77300.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77489

namespace LeftBound77493
def owner : Owner := ⟨.program ⟨214⟩, ⟨28284⟩⟩
def transferEvent : Nat := 77493
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 77491 .coefficient) (.predecessor 1 77492 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77491 .coefficient)
      LeftBound77486.bound (LeftBound77486.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77486.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77486.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77492 .coefficient)
      LeftBound5678.bound (LeftBound5678.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5679RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5678.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5678.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound77486.bound LeftBound5678.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77486.bound, LeftBound5678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound77486.actual selector witness) * (LeftBound5678.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77493

namespace LeftBound77494
def owner : Owner := ⟨.program ⟨214⟩, ⟨28284⟩⟩
def transferEvent : Nat := 77494
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩ [⟨.result 5675 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5675 .coefficient)
      LeftAuthority5674.bound (LeftAuthority5674.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6681⟩⟩) (rawTerms := some (Proof.Events022.exact5675RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5674.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5674.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5674.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5674.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound77494

namespace LeftBound77495
def owner : Owner := ⟨.program ⟨214⟩, ⟨28284⟩⟩
def transferEvent : Nat := 77495
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 77490 .summary) (.transfer 77494) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77490 .summary)
      LeftBound77489.bound (LeftBound77489.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28283⟩⟩) (rawTerms := some (Proof.Events302.exact77490RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77489.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 77494)
      LeftBound77494.bound (LeftBound77494.actual selector witness) := by
  exact .transfer (LeftBound77494.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound77489.bound LeftBound77494.bound
def bound : CoeffClass := .finite ⟨4742323242612988221224648704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77489.bound, LeftBound77494.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound77489.actual selector witness) * (LeftBound77494.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77495

namespace LeftBound77510
def owner : Owner := ⟨.program ⟨214⟩, ⟨28065⟩⟩
def transferEvent : Nat := 77510
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 77508 .coefficient) (.predecessor 1 77509 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77508 .coefficient)
      LeftBound69907.bound (LeftBound69907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69907.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77509 .coefficient)
      LeftAuthority77506.bound (LeftAuthority77506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77507RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77506.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77506.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound69907.bound LeftAuthority77506.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69907.bound, LeftAuthority77506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound69907.actual selector witness) * (LeftAuthority77506.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77510

namespace LeftBound77511
def owner : Owner := ⟨.program ⟨214⟩, ⟨28065⟩⟩
def transferEvent : Nat := 77511
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28063⟩⟩]⟩ [⟨.result 77507 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77507 .coefficient)
      LeftAuthority77506.bound (LeftAuthority77506.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28063⟩⟩) (rawTerms := some (Proof.Events302.exact77507RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77506.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77506.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority77506.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority77506.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound77511

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
