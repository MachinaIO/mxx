import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard675
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard721

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound105400
def owner : Owner := ⟨.program ⟨214⟩, ⟨16210⟩⟩
def transferEvent : Nat := 105400
def frameStart : Nat := 105356
def rule : BoundRule := .sum [.predecessor 0 105398 .coefficient, .predecessor 1 105399 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105398 .coefficient)
      LeftBound105383.bound (LeftBound105383.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound105383.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105399 .coefficient)
      LeftAuthority105396.bound (LeftAuthority105396.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority105396.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105383.bound, LeftAuthority105396.bound]
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105383.bound, LeftAuthority105396.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound105383.actual selector witness, LeftAuthority105396.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105400

namespace LeftBound105403
def owner : Owner := ⟨.program ⟨214⟩, ⟨16211⟩⟩
def transferEvent : Nat := 105403
def frameStart : Nat := 105356
def rule : BoundRule := .identity (.predecessor 0 105402 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105402 .coefficient)
      LeftBound105400.bound (LeftBound105400.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound105400.derived selector witness)

def rawBound : CoeffClass := LeftBound105400.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105400.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound105400.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound105403

namespace LeftBound105409
def owner : Owner := ⟨.program ⟨214⟩, ⟨16212⟩⟩
def transferEvent : Nat := 105409
def frameStart : Nat := 105356
def rule : BoundRule := .product (.predecessor 0 105407 .coefficient) (.predecessor 1 105408 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105407 .coefficient)
      LeftAuthority105405.bound (LeftAuthority105405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105405.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105405.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105408 .coefficient)
      LeftBound105403.bound (LeftBound105403.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105404RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105403.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105403.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority105405.bound LeftBound105403.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105405.bound, LeftBound105403.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority105405.actual selector witness) * (LeftBound105403.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105409

namespace LeftBound105417
def owner : Owner := ⟨.program ⟨214⟩, ⟨16213⟩⟩
def transferEvent : Nat := 105417
def frameStart : Nat := 105356
def rule : BoundRule := .sum [.predecessor 0 105415 .coefficient, .predecessor 1 105416 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105415 .coefficient)
      LeftAuthority105413.bound (LeftAuthority105413.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105414RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105413.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105413.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105416 .coefficient)
      LeftBound105409.bound (LeftBound105409.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105411RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105409.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105409.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority105413.bound, LeftBound105409.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105413.bound, LeftBound105409.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority105413.actual selector witness, LeftBound105409.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105417

namespace LeftBound105421
def owner : Owner := ⟨.program ⟨214⟩, ⟨28259⟩⟩
def transferEvent : Nat := 105421
def frameStart : Nat := 105356
def rule : BoundRule := .product (.predecessor 0 105419 .coefficient) (.predecessor 1 105420 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105419 .coefficient)
      LeftBound105417.bound (LeftBound105417.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105418RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105417.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105417.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105420 .coefficient)
      LeftAuthority105394.bound (LeftAuthority105394.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105395RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105394.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105394.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound105417.bound LeftAuthority105394.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105417.bound, LeftAuthority105394.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound105417.actual selector witness) * (LeftAuthority105394.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105421

namespace LeftBound105432
def owner : Owner := ⟨.program ⟨214⟩, ⟨17654⟩⟩
def transferEvent : Nat := 105432
def frameStart : Nat := 105356
def rule : BoundRule := .product (.predecessor 0 105430 .coefficient) (.predecessor 1 105431 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105430 .coefficient)
      LeftAuthority105405.bound (LeftAuthority105405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105405.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105405.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105431 .coefficient)
      LeftAuthority105428.bound (LeftAuthority105428.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105429RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105428.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105428.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority105405.bound LeftAuthority105428.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105405.bound, LeftAuthority105428.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority105405.actual selector witness) * (LeftAuthority105428.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105432

namespace LeftBound105440
def owner : Owner := ⟨.program ⟨214⟩, ⟨17655⟩⟩
def transferEvent : Nat := 105440
def frameStart : Nat := 105356
def rule : BoundRule := .sum [.predecessor 0 105438 .coefficient, .predecessor 1 105439 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105438 .coefficient)
      LeftAuthority105436.bound (LeftAuthority105436.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105437RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105436.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105436.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105439 .coefficient)
      LeftBound105432.bound (LeftBound105432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105434RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105432.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105432.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority105436.bound, LeftBound105432.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105436.bound, LeftBound105432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority105436.actual selector witness, LeftBound105432.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105440

namespace LeftBound105444
def owner : Owner := ⟨.program ⟨214⟩, ⟨28264⟩⟩
def transferEvent : Nat := 105444
def frameStart : Nat := 105356
def rule : BoundRule := .sum [.predecessor 0 105442 .coefficient, .predecessor 1 105443 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105442 .coefficient)
      LeftBound105440.bound (LeftBound105440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105440.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105440.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105443 .coefficient)
      LeftBound105421.bound (LeftBound105421.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105426RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105421.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105421.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105440.bound, LeftBound105421.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105440.bound, LeftBound105421.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound105440.actual selector witness, LeftBound105421.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105444

namespace LeftBound105457
def owner : Owner := ⟨.program ⟨214⟩, ⟨28261⟩⟩
def transferEvent : Nat := 105457
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 105455 .coefficient, .predecessor 1 105456 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105455 .coefficient)
      LeftBound105310.bound (LeftBound105310.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105454RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105310.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105310.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105456 .coefficient)
      LeftBound105293.bound (LeftBound105293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105300RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105293.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105310.bound, LeftBound105293.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105310.bound, LeftBound105293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound105310.actual selector witness, LeftBound105293.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105457

namespace LeftBound105460
def owner : Owner := ⟨.program ⟨214⟩, ⟨28261⟩⟩
def transferEvent : Nat := 105460
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 105454 .summary, .result 105300 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105454 .summary)
      LeftBound105312.bound (LeftBound105312.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21608⟩⟩) (rawTerms := some (Proof.Events411.exact105454RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105312.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105300 .summary)
      LeftBound105295.bound (LeftBound105295.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28260⟩⟩) (rawTerms := some (Proof.Events411.exact105300RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105295.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound105312.bound, LeftBound105295.bound]
def bound : CoeffClass := .finite ⟨1292180536164689260544, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105312.bound, LeftBound105295.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound105312.actual selector witness, LeftBound105295.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound105460

namespace LeftBound105464
def owner : Owner := ⟨.program ⟨214⟩, ⟨28262⟩⟩
def transferEvent : Nat := 105464
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 105462 .coefficient) (.predecessor 1 105463 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105462 .coefficient)
      LeftBound105457.bound (LeftBound105457.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105461RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105457.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105457.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105463 .coefficient)
      LeftBound5678.bound (LeftBound5678.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5679RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5678.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5678.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound105457.bound LeftBound5678.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105457.bound, LeftBound5678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound105457.actual selector witness) * (LeftBound5678.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105464

namespace LeftBound105465
def owner : Owner := ⟨.program ⟨214⟩, ⟨28262⟩⟩
def transferEvent : Nat := 105465
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
end LeftBound105465

namespace LeftBound105466
def owner : Owner := ⟨.program ⟨214⟩, ⟨28262⟩⟩
def transferEvent : Nat := 105466
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 105461 .summary) (.transfer 105465) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105461 .summary)
      LeftBound105460.bound (LeftBound105460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28261⟩⟩) (rawTerms := some (Proof.Events411.exact105461RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 105465)
      LeftBound105465.bound (LeftBound105465.actual selector witness) := by
  exact .transfer (LeftBound105465.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound105460.bound LeftBound105465.bound
def bound : CoeffClass := .finite ⟨4742323242612988221224648704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105460.bound, LeftBound105465.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound105460.actual selector witness) * (LeftBound105465.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105466

namespace LeftBound105481
def owner : Owner := ⟨.program ⟨214⟩, ⟨28043⟩⟩
def transferEvent : Nat := 105481
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 105479 .coefficient) (.predecessor 1 105480 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 105479 .coefficient)
      LeftBound98526.bound (LeftBound98526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98526.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 105480 .coefficient)
      LeftAuthority105477.bound (LeftAuthority105477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events412.exact105478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105477.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105477.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98526.bound LeftAuthority105477.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98526.bound, LeftAuthority105477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98526.actual selector witness) * (LeftAuthority105477.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105481

namespace LeftBound105482
def owner : Owner := ⟨.program ⟨214⟩, ⟨28043⟩⟩
def transferEvent : Nat := 105482
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28041⟩⟩]⟩ [⟨.result 105478 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 105478 .coefficient)
      LeftAuthority105477.bound (LeftAuthority105477.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28041⟩⟩) (rawTerms := some (Proof.Events412.exact105478RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority105477.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority105477.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority105477.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority105477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority105477.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound105482

namespace LeftBound105483
def owner : Owner := ⟨.program ⟨214⟩, ⟨28043⟩⟩
def transferEvent : Nat := 105483
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 98530 .summary) (.transfer 105482) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98530 .summary)
      LeftBound98529.bound (LeftBound98529.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26132⟩⟩) (rawTerms := some (Proof.Events384.exact98530RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98529.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 105482)
      LeftBound105482.bound (LeftBound105482.actual selector witness) := by
  exact .transfer (LeftBound105482.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98529.bound LeftBound105482.bound
def bound : CoeffClass := .finite ⟨1292113297018323992576, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98529.bound, LeftBound105482.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98529.actual selector witness) * (LeftBound105482.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound105483

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
