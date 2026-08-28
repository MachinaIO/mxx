import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard237
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard261

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound39237
def owner : Owner := ⟨.program ⟨214⟩, ⟨21987⟩⟩
def transferEvent : Nat := 39237
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 39236) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 39236)
      LeftBound39236.bound (LeftBound39236.actual selector witness) := by
  exact .transfer (LeftBound39236.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound39236.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound39236.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound39236.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39237

namespace LeftBound39332
def owner : Owner := ⟨.program ⟨214⟩, ⟨16390⟩⟩
def transferEvent : Nat := 39332
def frameStart : Nat := 39293
def rule : BoundRule := .identity (.predecessor 0 39331 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39331 .coefficient)
      LeftAuthority39329.bound (LeftAuthority39329.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39330RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39329.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39329.derived selector witness)

def rawBound : CoeffClass := LeftAuthority39329.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39329.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority39329.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound39332

namespace LeftBound39349
def owner : Owner := ⟨.program ⟨214⟩, ⟨16429⟩⟩
def transferEvent : Nat := 39349
def frameStart : Nat := 39293
def rule : BoundRule := .sum [.predecessor 0 39347 .coefficient, .predecessor 1 39348 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39347 .coefficient)
      LeftBound39332.bound (LeftBound39332.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound39332.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39348 .coefficient)
      LeftAuthority39345.bound (LeftAuthority39345.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority39345.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39332.bound, LeftAuthority39345.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39332.bound, LeftAuthority39345.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39332.actual selector witness, LeftAuthority39345.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39349

namespace LeftBound39352
def owner : Owner := ⟨.program ⟨214⟩, ⟨16430⟩⟩
def transferEvent : Nat := 39352
def frameStart : Nat := 39293
def rule : BoundRule := .identity (.predecessor 0 39351 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39351 .coefficient)
      LeftBound39349.bound (LeftBound39349.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound39349.derived selector witness)

def rawBound : CoeffClass := LeftBound39349.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39349.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound39349.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound39352

namespace LeftBound39358
def owner : Owner := ⟨.program ⟨214⟩, ⟨16431⟩⟩
def transferEvent : Nat := 39358
def frameStart : Nat := 39293
def rule : BoundRule := .product (.predecessor 0 39356 .coefficient) (.predecessor 1 39357 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39356 .coefficient)
      LeftAuthority39354.bound (LeftAuthority39354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39355RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39354.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39354.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39357 .coefficient)
      LeftBound39352.bound (LeftBound39352.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39353RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39352.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39352.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority39354.bound LeftBound39352.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39354.bound, LeftBound39352.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority39354.actual selector witness) * (LeftBound39352.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39358

namespace LeftBound39366
def owner : Owner := ⟨.program ⟨214⟩, ⟨16432⟩⟩
def transferEvent : Nat := 39366
def frameStart : Nat := 39293
def rule : BoundRule := .sum [.predecessor 0 39364 .coefficient, .predecessor 1 39365 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39364 .coefficient)
      LeftAuthority39362.bound (LeftAuthority39362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39363RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39362.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39362.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39365 .coefficient)
      LeftBound39358.bound (LeftBound39358.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39358.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39358.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority39362.bound, LeftBound39358.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39362.bound, LeftBound39358.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority39362.actual selector witness, LeftBound39358.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39366

namespace LeftBound39370
def owner : Owner := ⟨.program ⟨214⟩, ⟨28761⟩⟩
def transferEvent : Nat := 39370
def frameStart : Nat := 39293
def rule : BoundRule := .product (.predecessor 0 39368 .coefficient) (.predecessor 1 39369 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39368 .coefficient)
      LeftBound39366.bound (LeftBound39366.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39367RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39366.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39366.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39369 .coefficient)
      LeftAuthority39343.bound (LeftAuthority39343.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39344RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39343.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39343.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound39366.bound LeftAuthority39343.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39366.bound, LeftAuthority39343.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound39366.actual selector witness) * (LeftAuthority39343.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39370

namespace LeftBound39381
def owner : Owner := ⟨.program ⟨214⟩, ⟨17127⟩⟩
def transferEvent : Nat := 39381
def frameStart : Nat := 39293
def rule : BoundRule := .product (.predecessor 0 39379 .coefficient) (.predecessor 1 39380 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39379 .coefficient)
      LeftAuthority39354.bound (LeftAuthority39354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39355RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39354.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39354.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39380 .coefficient)
      LeftAuthority39377.bound (LeftAuthority39377.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39378RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39377.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39377.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority39354.bound LeftAuthority39377.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39354.bound, LeftAuthority39377.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority39354.actual selector witness) * (LeftAuthority39377.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39381

namespace LeftBound39389
def owner : Owner := ⟨.program ⟨214⟩, ⟨17128⟩⟩
def transferEvent : Nat := 39389
def frameStart : Nat := 39293
def rule : BoundRule := .sum [.predecessor 0 39387 .coefficient, .predecessor 1 39388 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39387 .coefficient)
      LeftAuthority39385.bound (LeftAuthority39385.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39385.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39388 .coefficient)
      LeftBound39381.bound (LeftBound39381.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39383RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39381.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39381.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority39385.bound, LeftBound39381.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39385.bound, LeftBound39381.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority39385.actual selector witness, LeftBound39381.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39389

namespace LeftBound39393
def owner : Owner := ⟨.program ⟨214⟩, ⟨28765⟩⟩
def transferEvent : Nat := 39393
def frameStart : Nat := 39293
def rule : BoundRule := .sum [.predecessor 0 39391 .coefficient, .predecessor 1 39392 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39391 .coefficient)
      LeftBound39389.bound (LeftBound39389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39390RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39389.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39389.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39392 .coefficient)
      LeftBound39370.bound (LeftBound39370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39375RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39370.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39370.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39389.bound, LeftBound39370.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39389.bound, LeftBound39370.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39389.actual selector witness, LeftBound39370.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39393

namespace LeftBound39406
def owner : Owner := ⟨.program ⟨214⟩, ⟨28763⟩⟩
def transferEvent : Nat := 39406
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 39404 .coefficient, .predecessor 1 39405 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39404 .coefficient)
      LeftBound39235.bound (LeftBound39235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39235.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39235.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39405 .coefficient)
      LeftBound39218.bound (LeftBound39218.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events153.exact39225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39218.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39218.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39235.bound, LeftBound39218.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39235.bound, LeftBound39218.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39235.actual selector witness, LeftBound39218.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39406

namespace LeftBound39409
def owner : Owner := ⟨.program ⟨214⟩, ⟨28763⟩⟩
def transferEvent : Nat := 39409
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 39403 .summary, .result 39225 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39403 .summary)
      LeftBound39237.bound (LeftBound39237.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21987⟩⟩) (rawTerms := some (Proof.Events153.exact39403RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39237.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39225 .summary)
      LeftBound39220.bound (LeftBound39220.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28762⟩⟩) (rawTerms := some (Proof.Events153.exact39225RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39220.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39237.bound, LeftBound39220.bound]
def bound : CoeffClass := .finite ⟨1292270185944771604480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39237.bound, LeftBound39220.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39237.actual selector witness, LeftBound39220.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39409

namespace LeftBound39433
def owner : Owner := ⟨.program ⟨214⟩, ⟨11780⟩⟩
def transferEvent : Nat := 39433
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 39431 .coefficient) (.predecessor 1 39432 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39431 .coefficient)
      LeftAuthority1750.bound (LeftAuthority1750.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1751RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1750.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1750.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39432 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1750.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1750.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1750.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound39433

namespace LeftBound39438
def owner : Owner := ⟨.program ⟨214⟩, ⟨7315⟩⟩
def transferEvent : Nat := 39438
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 39436 .coefficient) (.predecessor 1 39437 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39436 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39437 .coefficient)
      LeftBound9978.bound (LeftBound9978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9978.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound9978.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound9978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound9978.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39438

namespace LeftBound39443
def owner : Owner := ⟨.program ⟨214⟩, ⟨11781⟩⟩
def transferEvent : Nat := 39443
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 39441 .coefficient, .predecessor 1 39442 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39441 .coefficient)
      LeftBound39438.bound (LeftBound39438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39440RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39438.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39438.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39442 .coefficient)
      LeftBound39433.bound (LeftBound39433.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39435RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39433.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39433.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39438.bound, LeftBound39433.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39438.bound, LeftBound39433.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39438.actual selector witness, LeftBound39433.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39443

namespace LeftBound39447
def owner : Owner := ⟨.program ⟨214⟩, ⟨11782⟩⟩
def transferEvent : Nat := 39447
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 39445 .coefficient, .predecessor 1 39446 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39445 .coefficient)
      LeftBound39443.bound (LeftBound39443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events154.exact39444RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39443.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39443.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39446 .coefficient)
      LeftBound9970.bound (LeftBound9970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9971RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9970.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39443.bound, LeftBound9970.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39443.bound, LeftBound9970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39443.actual selector witness, LeftBound9970.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39447

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
