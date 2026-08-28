import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound65348
def owner : Owner := ⟨.program ⟨214⟩, ⟨10344⟩⟩
def transferEvent : Nat := 65348
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65343 .summary) (.transfer 65347) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65343 .summary)
      LeftBound65341.bound (LeftBound65341.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10343⟩⟩) (rawTerms := some (Proof.Events255.exact65343RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65341.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65347)
      LeftBound65347.bound (LeftBound65347.actual selector witness) := by
  exact .transfer (LeftBound65347.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound65341.bound LeftBound65347.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65341.bound, LeftBound65347.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound65341.actual selector witness) * (LeftBound65347.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65348

namespace LeftBound65356
def owner : Owner := ⟨.program ⟨214⟩, ⟨13349⟩⟩
def transferEvent : Nat := 65356
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 65354 .coefficient, .predecessor 1 65355 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65354 .coefficient)
      LeftBound65346.bound (LeftBound65346.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65353RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65346.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65346.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65355 .coefficient)
      LeftBound65318.bound (LeftBound65318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65318.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65346.bound, LeftBound65318.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65346.bound, LeftBound65318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65346.actual selector witness, LeftBound65318.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65356

namespace LeftBound65358
def owner : Owner := ⟨.program ⟨214⟩, ⟨13349⟩⟩
def transferEvent : Nat := 65358
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 65353 .summary, .result 65323 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65353 .summary)
      LeftBound65348.bound (LeftBound65348.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10344⟩⟩) (rawTerms := some (Proof.Events255.exact65353RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65348.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65323 .summary)
      LeftBound65320.bound (LeftBound65320.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13348⟩⟩) (rawTerms := some (Proof.Events255.exact65323RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65320.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65348.bound, LeftBound65320.bound]
def bound : CoeffClass := .finite ⟨95470336, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65348.bound, LeftBound65320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65348.actual selector witness, LeftBound65320.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65358

namespace LeftBound65362
def owner : Owner := ⟨.program ⟨214⟩, ⟨25754⟩⟩
def transferEvent : Nat := 65362
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 65360 .coefficient) (.predecessor 1 65361 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65360 .coefficient)
      LeftBound65356.bound (LeftBound65356.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65359RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65356.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65356.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65361 .coefficient)
      LeftAuthority65289.bound (LeftAuthority65289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65289.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65289.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound65356.bound LeftAuthority65289.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65356.bound, LeftAuthority65289.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound65356.actual selector witness) * (LeftAuthority65289.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65362

namespace LeftBound65363
def owner : Owner := ⟨.program ⟨214⟩, ⟨25754⟩⟩
def transferEvent : Nat := 65363
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25753⟩⟩]⟩ [⟨.result 65290 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65290 .coefficient)
      LeftAuthority65289.bound (LeftAuthority65289.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25753⟩⟩) (rawTerms := some (Proof.Events255.exact65290RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65289.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65289.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority65289.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65289.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority65289.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound65363

namespace LeftBound65364
def owner : Owner := ⟨.program ⟨214⟩, ⟨25754⟩⟩
def transferEvent : Nat := 65364
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65359 .summary) (.transfer 65363) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65359 .summary)
      LeftBound65358.bound (LeftBound65358.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13349⟩⟩) (rawTerms := some (Proof.Events255.exact65359RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65358.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65363)
      LeftBound65363.bound (LeftBound65363.actual selector witness) := by
  exact .transfer (LeftBound65363.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound65358.bound LeftBound65363.bound
def bound : CoeffClass := .finite ⟨350377660645376, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65358.bound, LeftBound65363.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound65358.actual selector witness) * (LeftBound65363.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65364

namespace LeftBound65375
def owner : Owner := ⟨.program ⟨214⟩, ⟨20246⟩⟩
def transferEvent : Nat := 65375
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 65373 .coefficient) (.value (.predecessor 1 65374 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65373 .coefficient)
      LeftAuthority65371.bound (LeftAuthority65371.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65372RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65371.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65371.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65374 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority65371.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65371.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority65371.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound65375

namespace LeftBound65379
def owner : Owner := ⟨.program ⟨214⟩, ⟨5534⟩⟩
def transferEvent : Nat := 65379
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 65377 .coefficient) (.predecessor 1 65378 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65377 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65378 .coefficient)
      LeftAuthority6549.bound (LeftAuthority6549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6549.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6549.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftAuthority6549.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftAuthority6549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftAuthority6549.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 16) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65379

namespace LeftBound65384
def owner : Owner := ⟨.program ⟨214⟩, ⟨5535⟩⟩
def transferEvent : Nat := 65384
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 65382 .coefficient, .predecessor 1 65383 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65382 .coefficient)
      LeftBound65379.bound (LeftBound65379.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65381RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65379.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65379.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65383 .coefficient)
      LeftAuthority6547.bound (LeftAuthority6547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6547.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6547.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65379.bound, LeftAuthority6547.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65379.bound, LeftAuthority6547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65379.actual selector witness, LeftAuthority6547.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65384

namespace LeftBound65385
def owner : Owner := ⟨.program ⟨214⟩, ⟨5535⟩⟩
def transferEvent : Nat := 65385
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22⟩⟩]⟩ [⟨.result 6548 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6548 .coefficient)
      LeftAuthority6547.bound (LeftAuthority6547.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22⟩⟩) (rawTerms := some (Proof.Events025.exact6548RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6547.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6547.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6547.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6547.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound65385

namespace LeftBound65390
def owner : Owner := ⟨.program ⟨214⟩, ⟨20247⟩⟩
def transferEvent : Nat := 65390
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 65388 .coefficient) (.predecessor 1 65389 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65388 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65389 .coefficient)
      LeftBound65375.bound (LeftBound65375.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65376RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65375.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65375.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound65375.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound65375.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound65375.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65390

namespace LeftBound65391
def owner : Owner := ⟨.program ⟨214⟩, ⟨20247⟩⟩
def transferEvent : Nat := 65391
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20244⟩⟩]⟩ [⟨.result 65372 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65372 .coefficient)
      LeftAuthority65371.bound (LeftAuthority65371.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20244⟩⟩) (rawTerms := some (Proof.Events255.exact65372RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65371.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65371.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority65371.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65371.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority65371.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound65391

namespace LeftBound65392
def owner : Owner := ⟨.program ⟨214⟩, ⟨20247⟩⟩
def transferEvent : Nat := 65392
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 65391) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65391)
      LeftBound65391.bound (LeftBound65391.actual selector witness) := by
  exact .transfer (LeftBound65391.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound65391.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound65391.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound65391.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65392

namespace LeftBound65471
def owner : Owner := ⟨.program ⟨214⟩, ⟨13343⟩⟩
def transferEvent : Nat := 65471
def frameStart : Nat := 65442
def rule : BoundRule := .product (.predecessor 0 65469 .coefficient) (.predecessor 1 65470 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65469 .coefficient)
      LeftAuthority65467.bound (LeftAuthority65467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65467.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65467.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65470 .coefficient)
      LeftAuthority65464.bound (LeftAuthority65464.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65465RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65464.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65464.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority65467.bound LeftAuthority65464.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65467.bound, LeftAuthority65464.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority65467.actual selector witness) * (LeftAuthority65464.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65471

namespace LeftBound65475
def owner : Owner := ⟨.program ⟨214⟩, ⟨13344⟩⟩
def transferEvent : Nat := 65475
def frameStart : Nat := 65442
def rule : BoundRule := .identity (.predecessor 0 65474 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65474 .coefficient)
      LeftBound65471.bound (LeftBound65471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65473RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65471.derived selector witness)

def rawBound : CoeffClass := LeftBound65471.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound65471.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound65475

namespace LeftBound65492
def owner : Owner := ⟨.program ⟨214⟩, ⟨13442⟩⟩
def transferEvent : Nat := 65492
def frameStart : Nat := 65442
def rule : BoundRule := .sum [.predecessor 0 65490 .coefficient, .predecessor 1 65491 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65490 .coefficient)
      LeftBound65475.bound (LeftBound65475.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound65475.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65491 .coefficient)
      LeftAuthority65488.bound (LeftAuthority65488.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority65488.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65475.bound, LeftAuthority65488.bound]
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65475.bound, LeftAuthority65488.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65475.actual selector witness, LeftAuthority65488.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65492

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
