import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard448

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound66356
def owner : Owner := ⟨.program ⟨214⟩, ⟨20103⟩⟩
def transferEvent : Nat := 66356
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 66355) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 66355)
      LeftBound66355.bound (LeftBound66355.actual selector witness) := by
  exact .transfer (LeftBound66355.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound66355.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound66355.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound66355.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66356

namespace LeftBound66435
def owner : Owner := ⟨.program ⟨214⟩, ⟨12951⟩⟩
def transferEvent : Nat := 66435
def frameStart : Nat := 66406
def rule : BoundRule := .product (.predecessor 0 66433 .coefficient) (.predecessor 1 66434 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66433 .coefficient)
      LeftAuthority66431.bound (LeftAuthority66431.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66432RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66431.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66431.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66434 .coefficient)
      LeftAuthority66428.bound (LeftAuthority66428.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66429RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66428.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66428.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority66431.bound LeftAuthority66428.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66431.bound, LeftAuthority66428.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority66431.actual selector witness) * (LeftAuthority66428.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66435

namespace LeftBound66439
def owner : Owner := ⟨.program ⟨214⟩, ⟨12952⟩⟩
def transferEvent : Nat := 66439
def frameStart : Nat := 66406
def rule : BoundRule := .identity (.predecessor 0 66438 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66438 .coefficient)
      LeftBound66435.bound (LeftBound66435.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66437RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66435.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66435.derived selector witness)

def rawBound : CoeffClass := LeftBound66435.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66435.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound66435.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound66439

namespace LeftBound66456
def owner : Owner := ⟨.program ⟨214⟩, ⟨13050⟩⟩
def transferEvent : Nat := 66456
def frameStart : Nat := 66406
def rule : BoundRule := .sum [.predecessor 0 66454 .coefficient, .predecessor 1 66455 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66454 .coefficient)
      LeftBound66439.bound (LeftBound66439.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound66439.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66455 .coefficient)
      LeftAuthority66452.bound (LeftAuthority66452.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority66452.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66439.bound, LeftAuthority66452.bound]
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66439.bound, LeftAuthority66452.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66439.actual selector witness, LeftAuthority66452.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66456

namespace LeftBound66459
def owner : Owner := ⟨.program ⟨214⟩, ⟨13051⟩⟩
def transferEvent : Nat := 66459
def frameStart : Nat := 66406
def rule : BoundRule := .identity (.predecessor 0 66458 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66458 .coefficient)
      LeftBound66456.bound (LeftBound66456.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound66456.derived selector witness)

def rawBound : CoeffClass := LeftBound66456.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound66456.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound66459

namespace LeftBound66465
def owner : Owner := ⟨.program ⟨214⟩, ⟨13052⟩⟩
def transferEvent : Nat := 66465
def frameStart : Nat := 66406
def rule : BoundRule := .product (.predecessor 0 66463 .coefficient) (.predecessor 1 66464 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66463 .coefficient)
      LeftAuthority66461.bound (LeftAuthority66461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66461.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66461.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66464 .coefficient)
      LeftBound66459.bound (LeftBound66459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66459.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority66461.bound LeftBound66459.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66461.bound, LeftBound66459.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority66461.actual selector witness) * (LeftBound66459.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66465

namespace LeftBound66481
def owner : Owner := ⟨.program ⟨214⟩, ⟨7877⟩⟩
def transferEvent : Nat := 66481
def frameStart : Nat := 66406
def rule : BoundRule := .scale (.predecessor 0 66479 .coefficient) (.value (.predecessor 1 66480 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66479 .coefficient)
      LeftAuthority66477.bound (LeftAuthority66477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66477.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66477.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66480 .coefficient)
      LeftAuthority66468.bound (LeftAuthority66468.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority66468.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority66477.bound LeftAuthority66468.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66477.bound, LeftAuthority66468.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority66477.actual selector witness) * (LeftAuthority66468.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound66481

namespace LeftBound66484
def owner : Owner := ⟨.program ⟨214⟩, ⟨6768⟩⟩
def transferEvent : Nat := 66484
def frameStart : Nat := 66406
def rule : BoundRule := .identity (.predecessor 0 66483 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66483 .coefficient)
      LeftAuthority66471.bound (LeftAuthority66471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66471.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66471.derived selector witness)

def rawBound : CoeffClass := LeftAuthority66471.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority66471.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound66484

namespace LeftBound66488
def owner : Owner := ⟨.program ⟨214⟩, ⟨7878⟩⟩
def transferEvent : Nat := 66488
def frameStart : Nat := 66406
def rule : BoundRule := .product (.predecessor 0 66486 .coefficient) (.predecessor 1 66487 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66486 .coefficient)
      LeftBound66484.bound (LeftBound66484.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66485RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66484.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66484.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66487 .coefficient)
      LeftBound66481.bound (LeftBound66481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66481.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66481.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66484.bound LeftBound66481.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66484.bound, LeftBound66481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66484.actual selector witness) * (LeftBound66481.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66488

namespace LeftBound66493
def owner : Owner := ⟨.program ⟨214⟩, ⟨13053⟩⟩
def transferEvent : Nat := 66493
def frameStart : Nat := 66406
def rule : BoundRule := .sum [.predecessor 0 66491 .coefficient, .predecessor 1 66492 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66491 .coefficient)
      LeftBound66488.bound (LeftBound66488.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66488.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66488.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66492 .coefficient)
      LeftBound66465.bound (LeftBound66465.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66467RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66465.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66488.bound, LeftBound66465.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66488.bound, LeftBound66465.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66488.actual selector witness, LeftBound66465.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66493

namespace LeftBound66497
def owner : Owner := ⟨.program ⟨214⟩, ⟨25602⟩⟩
def transferEvent : Nat := 66497
def frameStart : Nat := 66406
def rule : BoundRule := .product (.predecessor 0 66495 .coefficient) (.predecessor 1 66496 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66495 .coefficient)
      LeftBound66493.bound (LeftBound66493.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66493.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66496 .coefficient)
      LeftAuthority66450.bound (LeftAuthority66450.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66451RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66450.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66450.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66493.bound LeftAuthority66450.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66493.bound, LeftAuthority66450.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66493.actual selector witness) * (LeftAuthority66450.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66497

namespace LeftBound66508
def owner : Owner := ⟨.program ⟨214⟩, ⟨16750⟩⟩
def transferEvent : Nat := 66508
def frameStart : Nat := 66406
def rule : BoundRule := .product (.predecessor 0 66506 .coefficient) (.predecessor 1 66507 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66506 .coefficient)
      LeftAuthority66461.bound (LeftAuthority66461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66461.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66461.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66507 .coefficient)
      LeftAuthority66504.bound (LeftAuthority66504.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66505RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66504.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66504.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority66461.bound LeftAuthority66504.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66461.bound, LeftAuthority66504.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority66461.actual selector witness) * (LeftAuthority66504.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66508

namespace LeftBound66516
def owner : Owner := ⟨.program ⟨214⟩, ⟨16751⟩⟩
def transferEvent : Nat := 66516
def frameStart : Nat := 66406
def rule : BoundRule := .sum [.predecessor 0 66514 .coefficient, .predecessor 1 66515 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66514 .coefficient)
      LeftAuthority66512.bound (LeftAuthority66512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66512.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66512.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66515 .coefficient)
      LeftBound66508.bound (LeftBound66508.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66508.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority66512.bound, LeftBound66508.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66512.bound, LeftBound66508.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority66512.actual selector witness, LeftBound66508.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66516

namespace LeftBound66520
def owner : Owner := ⟨.program ⟨214⟩, ⟨25603⟩⟩
def transferEvent : Nat := 66520
def frameStart : Nat := 66406
def rule : BoundRule := .sum [.predecessor 0 66518 .coefficient, .predecessor 1 66519 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66518 .coefficient)
      LeftBound66516.bound (LeftBound66516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66516.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66519 .coefficient)
      LeftBound66497.bound (LeftBound66497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66502RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66497.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66497.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66516.bound, LeftBound66497.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66516.bound, LeftBound66497.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66516.actual selector witness, LeftBound66497.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66520

namespace LeftBound66533
def owner : Owner := ⟨.program ⟨214⟩, ⟨25601⟩⟩
def transferEvent : Nat := 66533
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 66531 .coefficient, .predecessor 1 66532 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66531 .coefficient)
      LeftBound66354.bound (LeftBound66354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66354.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66532 .coefficient)
      LeftBound66337.bound (LeftBound66337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66344RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66337.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66337.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66354.bound, LeftBound66337.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66354.bound, LeftBound66337.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66354.actual selector witness, LeftBound66337.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66533

namespace LeftBound66536
def owner : Owner := ⟨.program ⟨214⟩, ⟨25601⟩⟩
def transferEvent : Nat := 66536
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 66530 .summary, .result 66344 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66530 .summary)
      LeftBound66356.bound (LeftBound66356.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20103⟩⟩) (rawTerms := some (Proof.Events259.exact66530RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66356.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66344 .summary)
      LeftBound66339.bound (LeftBound66339.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25600⟩⟩) (rawTerms := some (Proof.Events259.exact66344RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66339.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66356.bound, LeftBound66339.bound]
def bound : CoeffClass := .finite ⟨352164536528896, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66356.bound, LeftBound66339.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66356.actual selector witness, LeftBound66339.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66536

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
