import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard285

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound42408
def owner : Owner := ⟨.program ⟨214⟩, ⟨19323⟩⟩
def transferEvent : Nat := 42408
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 42407) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 42407)
      LeftBound42407.bound (LeftBound42407.actual selector witness) := by
  exact .transfer (LeftBound42407.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound42407.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound42407.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound42407.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42408

namespace LeftBound42487
def owner : Owner := ⟨.program ⟨214⟩, ⟨13575⟩⟩
def transferEvent : Nat := 42487
def frameStart : Nat := 42458
def rule : BoundRule := .product (.predecessor 0 42485 .coefficient) (.predecessor 1 42486 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42485 .coefficient)
      LeftAuthority42483.bound (LeftAuthority42483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42483.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42483.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42486 .coefficient)
      LeftAuthority42480.bound (LeftAuthority42480.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42481RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42480.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42480.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority42483.bound LeftAuthority42480.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42483.bound, LeftAuthority42480.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority42483.actual selector witness) * (LeftAuthority42480.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42487

namespace LeftBound42491
def owner : Owner := ⟨.program ⟨214⟩, ⟨13576⟩⟩
def transferEvent : Nat := 42491
def frameStart : Nat := 42458
def rule : BoundRule := .identity (.predecessor 0 42490 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42490 .coefficient)
      LeftBound42487.bound (LeftBound42487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42489RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42487.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42487.derived selector witness)

def rawBound : CoeffClass := LeftBound42487.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42487.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound42487.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound42491

namespace LeftBound42508
def owner : Owner := ⟨.program ⟨214⟩, ⟨13671⟩⟩
def transferEvent : Nat := 42508
def frameStart : Nat := 42458
def rule : BoundRule := .sum [.predecessor 0 42506 .coefficient, .predecessor 1 42507 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42506 .coefficient)
      LeftBound42491.bound (LeftBound42491.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound42491.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42507 .coefficient)
      LeftAuthority42504.bound (LeftAuthority42504.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority42504.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42491.bound, LeftAuthority42504.bound]
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42491.bound, LeftAuthority42504.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42491.actual selector witness, LeftAuthority42504.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42508

namespace LeftBound42511
def owner : Owner := ⟨.program ⟨214⟩, ⟨13672⟩⟩
def transferEvent : Nat := 42511
def frameStart : Nat := 42458
def rule : BoundRule := .identity (.predecessor 0 42510 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42510 .coefficient)
      LeftBound42508.bound (LeftBound42508.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound42508.derived selector witness)

def rawBound : CoeffClass := LeftBound42508.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42508.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound42508.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound42511

namespace LeftBound42517
def owner : Owner := ⟨.program ⟨214⟩, ⟨13673⟩⟩
def transferEvent : Nat := 42517
def frameStart : Nat := 42458
def rule : BoundRule := .product (.predecessor 0 42515 .coefficient) (.predecessor 1 42516 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42515 .coefficient)
      LeftAuthority42513.bound (LeftAuthority42513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42514RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42513.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42513.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42516 .coefficient)
      LeftBound42511.bound (LeftBound42511.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42511.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42511.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority42513.bound LeftBound42511.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42513.bound, LeftBound42511.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority42513.actual selector witness) * (LeftBound42511.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42517

namespace LeftBound42533
def owner : Owner := ⟨.program ⟨214⟩, ⟨7844⟩⟩
def transferEvent : Nat := 42533
def frameStart : Nat := 42458
def rule : BoundRule := .scale (.predecessor 0 42531 .coefficient) (.value (.predecessor 1 42532 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42531 .coefficient)
      LeftAuthority42529.bound (LeftAuthority42529.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42529.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42529.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42532 .coefficient)
      LeftAuthority42520.bound (LeftAuthority42520.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority42520.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority42529.bound LeftAuthority42520.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42529.bound, LeftAuthority42520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority42529.actual selector witness) * (LeftAuthority42520.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound42533

namespace LeftBound42536
def owner : Owner := ⟨.program ⟨214⟩, ⟨6793⟩⟩
def transferEvent : Nat := 42536
def frameStart : Nat := 42458
def rule : BoundRule := .identity (.predecessor 0 42535 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42535 .coefficient)
      LeftAuthority42523.bound (LeftAuthority42523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42524RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42523.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42523.derived selector witness)

def rawBound : CoeffClass := LeftAuthority42523.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42523.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority42523.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound42536

namespace LeftBound42540
def owner : Owner := ⟨.program ⟨214⟩, ⟨7845⟩⟩
def transferEvent : Nat := 42540
def frameStart : Nat := 42458
def rule : BoundRule := .product (.predecessor 0 42538 .coefficient) (.predecessor 1 42539 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42538 .coefficient)
      LeftBound42536.bound (LeftBound42536.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42537RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42536.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42536.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42539 .coefficient)
      LeftBound42533.bound (LeftBound42533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42534RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42533.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42533.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound42536.bound LeftBound42533.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42536.bound, LeftBound42533.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound42536.actual selector witness) * (LeftBound42533.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42540

namespace LeftBound42545
def owner : Owner := ⟨.program ⟨214⟩, ⟨13674⟩⟩
def transferEvent : Nat := 42545
def frameStart : Nat := 42458
def rule : BoundRule := .sum [.predecessor 0 42543 .coefficient, .predecessor 1 42544 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42543 .coefficient)
      LeftBound42540.bound (LeftBound42540.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42542RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42540.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42540.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42544 .coefficient)
      LeftBound42517.bound (LeftBound42517.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42519RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42517.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42517.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42540.bound, LeftBound42517.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42540.bound, LeftBound42517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42540.actual selector witness, LeftBound42517.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42545

namespace LeftBound42549
def owner : Owner := ⟨.program ⟨214⟩, ⟨25848⟩⟩
def transferEvent : Nat := 42549
def frameStart : Nat := 42458
def rule : BoundRule := .product (.predecessor 0 42547 .coefficient) (.predecessor 1 42548 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42547 .coefficient)
      LeftBound42545.bound (LeftBound42545.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42546RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42545.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42545.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42548 .coefficient)
      LeftAuthority42502.bound (LeftAuthority42502.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42502.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42502.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound42545.bound LeftAuthority42502.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42545.bound, LeftAuthority42502.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound42545.actual selector witness) * (LeftAuthority42502.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42549

namespace LeftBound42560
def owner : Owner := ⟨.program ⟨214⟩, ⟨15593⟩⟩
def transferEvent : Nat := 42560
def frameStart : Nat := 42458
def rule : BoundRule := .product (.predecessor 0 42558 .coefficient) (.predecessor 1 42559 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42558 .coefficient)
      LeftAuthority42513.bound (LeftAuthority42513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42514RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42513.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42513.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42559 .coefficient)
      LeftAuthority42556.bound (LeftAuthority42556.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42557RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42556.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42556.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority42513.bound LeftAuthority42556.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42513.bound, LeftAuthority42556.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority42513.actual selector witness) * (LeftAuthority42556.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42560

namespace LeftBound42568
def owner : Owner := ⟨.program ⟨214⟩, ⟨15594⟩⟩
def transferEvent : Nat := 42568
def frameStart : Nat := 42458
def rule : BoundRule := .sum [.predecessor 0 42566 .coefficient, .predecessor 1 42567 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42566 .coefficient)
      LeftAuthority42564.bound (LeftAuthority42564.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42565RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42564.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42564.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42567 .coefficient)
      LeftBound42560.bound (LeftBound42560.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42562RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42560.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42560.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority42564.bound, LeftBound42560.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42564.bound, LeftBound42560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority42564.actual selector witness, LeftBound42560.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42568

namespace LeftBound42572
def owner : Owner := ⟨.program ⟨214⟩, ⟨25849⟩⟩
def transferEvent : Nat := 42572
def frameStart : Nat := 42458
def rule : BoundRule := .sum [.predecessor 0 42570 .coefficient, .predecessor 1 42571 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42570 .coefficient)
      LeftBound42568.bound (LeftBound42568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42569RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42568.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42568.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42571 .coefficient)
      LeftBound42549.bound (LeftBound42549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42554RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42549.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42549.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42568.bound, LeftBound42549.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42568.bound, LeftBound42549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42568.actual selector witness, LeftBound42549.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42572

namespace LeftBound42585
def owner : Owner := ⟨.program ⟨214⟩, ⟨25847⟩⟩
def transferEvent : Nat := 42585
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 42583 .coefficient, .predecessor 1 42584 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42583 .coefficient)
      LeftBound42406.bound (LeftBound42406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events166.exact42582RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42406.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42406.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42584 .coefficient)
      LeftBound42389.bound (LeftBound42389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42389.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42389.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42406.bound, LeftBound42389.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42406.bound, LeftBound42389.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42406.actual selector witness, LeftBound42389.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42585

namespace LeftBound42588
def owner : Owner := ⟨.program ⟨214⟩, ⟨25847⟩⟩
def transferEvent : Nat := 42588
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 42582 .summary, .result 42396 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42582 .summary)
      LeftBound42408.bound (LeftBound42408.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19323⟩⟩) (rawTerms := some (Proof.Events166.exact42582RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound42408.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42396 .summary)
      LeftBound42391.bound (LeftBound42391.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25846⟩⟩) (rawTerms := some (Proof.Events165.exact42396RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound42391.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42408.bound, LeftBound42391.bound]
def bound : CoeffClass := .finite ⟨352036291489792, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42408.bound, LeftBound42391.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42408.actual selector witness, LeftBound42391.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42588

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
