import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard464

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound68599
def owner : Owner := ⟨.program ⟨214⟩, ⟨16417⟩⟩
def transferEvent : Nat := 68599
def frameStart : Nat := 68543
def rule : BoundRule := .sum [.predecessor 0 68597 .coefficient, .predecessor 1 68598 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68597 .coefficient)
      LeftBound68582.bound (LeftBound68582.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound68582.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68598 .coefficient)
      LeftAuthority68595.bound (LeftAuthority68595.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority68595.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68582.bound, LeftAuthority68595.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68582.bound, LeftAuthority68595.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68582.actual selector witness, LeftAuthority68595.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68599

namespace LeftBound68602
def owner : Owner := ⟨.program ⟨214⟩, ⟨16418⟩⟩
def transferEvent : Nat := 68602
def frameStart : Nat := 68543
def rule : BoundRule := .identity (.predecessor 0 68601 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68601 .coefficient)
      LeftBound68599.bound (LeftBound68599.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound68599.derived selector witness)

def rawBound : CoeffClass := LeftBound68599.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68599.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound68599.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound68602

namespace LeftBound68608
def owner : Owner := ⟨.program ⟨214⟩, ⟨16419⟩⟩
def transferEvent : Nat := 68608
def frameStart : Nat := 68543
def rule : BoundRule := .product (.predecessor 0 68606 .coefficient) (.predecessor 1 68607 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68606 .coefficient)
      LeftAuthority68604.bound (LeftAuthority68604.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68605RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68604.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68604.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68607 .coefficient)
      LeftBound68602.bound (LeftBound68602.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68603RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68602.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68602.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority68604.bound LeftBound68602.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68604.bound, LeftBound68602.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority68604.actual selector witness) * (LeftBound68602.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68608

namespace LeftBound68616
def owner : Owner := ⟨.program ⟨214⟩, ⟨16420⟩⟩
def transferEvent : Nat := 68616
def frameStart : Nat := 68543
def rule : BoundRule := .sum [.predecessor 0 68614 .coefficient, .predecessor 1 68615 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68614 .coefficient)
      LeftAuthority68612.bound (LeftAuthority68612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68613RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68612.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68612.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68615 .coefficient)
      LeftBound68608.bound (LeftBound68608.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68610RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68608.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68608.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority68612.bound, LeftBound68608.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68612.bound, LeftBound68608.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority68612.actual selector witness, LeftBound68608.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68616

namespace LeftBound68620
def owner : Owner := ⟨.program ⟨214⟩, ⟨28722⟩⟩
def transferEvent : Nat := 68620
def frameStart : Nat := 68543
def rule : BoundRule := .product (.predecessor 0 68618 .coefficient) (.predecessor 1 68619 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68618 .coefficient)
      LeftBound68616.bound (LeftBound68616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68617RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68616.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68616.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68619 .coefficient)
      LeftAuthority68593.bound (LeftAuthority68593.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68594RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68593.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68593.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound68616.bound LeftAuthority68593.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68616.bound, LeftAuthority68593.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound68616.actual selector witness) * (LeftAuthority68593.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68620

namespace LeftBound68631
def owner : Owner := ⟨.program ⟨214⟩, ⟨17118⟩⟩
def transferEvent : Nat := 68631
def frameStart : Nat := 68543
def rule : BoundRule := .product (.predecessor 0 68629 .coefficient) (.predecessor 1 68630 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68629 .coefficient)
      LeftAuthority68604.bound (LeftAuthority68604.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68605RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68604.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68604.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68630 .coefficient)
      LeftAuthority68627.bound (LeftAuthority68627.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68627.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68627.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority68604.bound LeftAuthority68627.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68604.bound, LeftAuthority68627.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority68604.actual selector witness) * (LeftAuthority68627.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68631

namespace LeftBound68639
def owner : Owner := ⟨.program ⟨214⟩, ⟨17119⟩⟩
def transferEvent : Nat := 68639
def frameStart : Nat := 68543
def rule : BoundRule := .sum [.predecessor 0 68637 .coefficient, .predecessor 1 68638 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68637 .coefficient)
      LeftAuthority68635.bound (LeftAuthority68635.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68635.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68635.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68638 .coefficient)
      LeftBound68631.bound (LeftBound68631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68633RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68631.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68631.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority68635.bound, LeftBound68631.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68635.bound, LeftBound68631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority68635.actual selector witness, LeftBound68631.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68639

namespace LeftBound68643
def owner : Owner := ⟨.program ⟨214⟩, ⟨28726⟩⟩
def transferEvent : Nat := 68643
def frameStart : Nat := 68543
def rule : BoundRule := .sum [.predecessor 0 68641 .coefficient, .predecessor 1 68642 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68641 .coefficient)
      LeftBound68639.bound (LeftBound68639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68640RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68639.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68639.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68642 .coefficient)
      LeftBound68620.bound (LeftBound68620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68620.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68620.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68639.bound, LeftBound68620.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68639.bound, LeftBound68620.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68639.actual selector witness, LeftBound68620.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68643

namespace LeftBound68656
def owner : Owner := ⟨.program ⟨214⟩, ⟨28724⟩⟩
def transferEvent : Nat := 68656
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 68654 .coefficient, .predecessor 1 68655 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68654 .coefficient)
      LeftBound68485.bound (LeftBound68485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68485.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68485.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68655 .coefficient)
      LeftBound68468.bound (LeftBound68468.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68468.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68468.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68485.bound, LeftBound68468.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68485.bound, LeftBound68468.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68485.actual selector witness, LeftBound68468.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68656

namespace LeftBound68659
def owner : Owner := ⟨.program ⟨214⟩, ⟨28724⟩⟩
def transferEvent : Nat := 68659
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 68653 .summary, .result 68475 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68653 .summary)
      LeftBound68487.bound (LeftBound68487.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21975⟩⟩) (rawTerms := some (Proof.Events268.exact68653RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68487.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68475 .summary)
      LeftBound68470.bound (LeftBound68470.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28723⟩⟩) (rawTerms := some (Proof.Events267.exact68475RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68470.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68487.bound, LeftBound68470.bound]
def bound : CoeffClass := .finite ⟨1292270185944771604480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68487.bound, LeftBound68470.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68487.actual selector witness, LeftBound68470.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68659

namespace LeftBound68683
def owner : Owner := ⟨.program ⟨214⟩, ⟨11756⟩⟩
def transferEvent : Nat := 68683
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 68681 .coefficient) (.predecessor 1 68682 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68681 .coefficient)
      LeftAuthority3246.bound (LeftAuthority3246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3247RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3246.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68682 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3246.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3246.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3246.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound68683

namespace LeftBound68688
def owner : Owner := ⟨.program ⟨214⟩, ⟨7201⟩⟩
def transferEvent : Nat := 68688
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 68686 .coefficient) (.predecessor 1 68687 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68686 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68687 .coefficient)
      LeftBound9978.bound (LeftBound9978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9978.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound9978.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound9978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound9978.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68688

namespace LeftBound68693
def owner : Owner := ⟨.program ⟨214⟩, ⟨11757⟩⟩
def transferEvent : Nat := 68693
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 68691 .coefficient, .predecessor 1 68692 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68691 .coefficient)
      LeftBound68688.bound (LeftBound68688.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68690RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68688.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68688.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68692 .coefficient)
      LeftBound68683.bound (LeftBound68683.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68685RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68683.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68683.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68688.bound, LeftBound68683.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68688.bound, LeftBound68683.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68688.actual selector witness, LeftBound68683.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68693

namespace LeftBound68697
def owner : Owner := ⟨.program ⟨214⟩, ⟨11758⟩⟩
def transferEvent : Nat := 68697
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 68695 .coefficient, .predecessor 1 68696 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68695 .coefficient)
      LeftBound68693.bound (LeftBound68693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68694RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68693.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68693.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68696 .coefficient)
      LeftBound9970.bound (LeftBound9970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9971RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9970.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68693.bound, LeftBound9970.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68693.bound, LeftBound9970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68693.actual selector witness, LeftBound9970.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68697

namespace LeftBound68698
def owner : Owner := ⟨.program ⟨214⟩, ⟨11758⟩⟩
def transferEvent : Nat := 68698
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
end LeftBound68698

namespace LeftBound68703
def owner : Owner := ⟨.program ⟨214⟩, ⟨11759⟩⟩
def transferEvent : Nat := 68703
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 68701 .coefficient) (.predecessor 1 68702 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68701 .coefficient)
      LeftBound68697.bound (LeftBound68697.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68697.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68697.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68702 .coefficient)
      LeftAuthority3249.bound (LeftAuthority3249.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3249.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3249.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound68697.bound LeftAuthority3249.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68697.bound, LeftAuthority3249.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound68697.actual selector witness) * (LeftAuthority3249.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68703

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
