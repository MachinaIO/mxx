import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard082

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound13276
def owner : Owner := ⟨.program ⟨214⟩, ⟨20986⟩⟩
def transferEvent : Nat := 13276
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 13274 .coefficient) (.value (.predecessor 1 13275 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13274 .coefficient)
      LeftAuthority13272.bound (LeftAuthority13272.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13273RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13272.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13272.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13275 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority13272.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13272.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority13272.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound13276

namespace LeftBound13280
def owner : Owner := ⟨.program ⟨214⟩, ⟨20987⟩⟩
def transferEvent : Nat := 13280
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 13278 .coefficient) (.predecessor 1 13279 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13278 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13279 .coefficient)
      LeftBound13276.bound (LeftBound13276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13277RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13276.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13276.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound13276.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound13276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound13276.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13280

namespace LeftBound13281
def owner : Owner := ⟨.program ⟨214⟩, ⟨20987⟩⟩
def transferEvent : Nat := 13281
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20984⟩⟩]⟩ [⟨.result 13273 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13273 .coefficient)
      LeftAuthority13272.bound (LeftAuthority13272.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20984⟩⟩) (rawTerms := some (Proof.Events051.exact13273RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13272.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13272.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority13272.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13272.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority13272.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound13281

namespace LeftBound13282
def owner : Owner := ⟨.program ⟨214⟩, ⟨20987⟩⟩
def transferEvent : Nat := 13282
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 13281) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 13281)
      LeftBound13281.bound (LeftBound13281.actual selector witness) := by
  exact .transfer (LeftBound13281.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound13281.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound13281.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound13281.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13282

namespace LeftBound13377
def owner : Owner := ⟨.program ⟨214⟩, ⟨15600⟩⟩
def transferEvent : Nat := 13377
def frameStart : Nat := 13338
def rule : BoundRule := .identity (.predecessor 0 13376 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13376 .coefficient)
      LeftAuthority13374.bound (LeftAuthority13374.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13375RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13374.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13374.derived selector witness)

def rawBound : CoeffClass := LeftAuthority13374.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13374.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority13374.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound13377

namespace LeftBound13394
def owner : Owner := ⟨.program ⟨214⟩, ⟨15674⟩⟩
def transferEvent : Nat := 13394
def frameStart : Nat := 13338
def rule : BoundRule := .sum [.predecessor 0 13392 .coefficient, .predecessor 1 13393 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13392 .coefficient)
      LeftBound13377.bound (LeftBound13377.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound13377.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13393 .coefficient)
      LeftAuthority13390.bound (LeftAuthority13390.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority13390.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13377.bound, LeftAuthority13390.bound]
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13377.bound, LeftAuthority13390.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13377.actual selector witness, LeftAuthority13390.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13394

namespace LeftBound13397
def owner : Owner := ⟨.program ⟨214⟩, ⟨15675⟩⟩
def transferEvent : Nat := 13397
def frameStart : Nat := 13338
def rule : BoundRule := .identity (.predecessor 0 13396 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13396 .coefficient)
      LeftBound13394.bound (LeftBound13394.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound13394.derived selector witness)

def rawBound : CoeffClass := LeftBound13394.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13394.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound13394.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound13397

namespace LeftBound13403
def owner : Owner := ⟨.program ⟨214⟩, ⟨15676⟩⟩
def transferEvent : Nat := 13403
def frameStart : Nat := 13338
def rule : BoundRule := .product (.predecessor 0 13401 .coefficient) (.predecessor 1 13402 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13401 .coefficient)
      LeftAuthority13399.bound (LeftAuthority13399.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13400RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13399.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13399.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13402 .coefficient)
      LeftBound13397.bound (LeftBound13397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13397.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority13399.bound LeftBound13397.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13399.bound, LeftBound13397.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority13399.actual selector witness) * (LeftBound13397.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13403

namespace LeftBound13411
def owner : Owner := ⟨.program ⟨214⟩, ⟨15677⟩⟩
def transferEvent : Nat := 13411
def frameStart : Nat := 13338
def rule : BoundRule := .sum [.predecessor 0 13409 .coefficient, .predecessor 1 13410 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13409 .coefficient)
      LeftAuthority13407.bound (LeftAuthority13407.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13408RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13407.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13407.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13410 .coefficient)
      LeftBound13403.bound (LeftBound13403.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13405RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13403.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13403.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority13407.bound, LeftBound13403.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13407.bound, LeftBound13403.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority13407.actual selector witness, LeftBound13403.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13411

namespace LeftBound13415
def owner : Owner := ⟨.program ⟨214⟩, ⟨27268⟩⟩
def transferEvent : Nat := 13415
def frameStart : Nat := 13338
def rule : BoundRule := .product (.predecessor 0 13413 .coefficient) (.predecessor 1 13414 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13413 .coefficient)
      LeftBound13411.bound (LeftBound13411.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13412RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13411.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13411.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13414 .coefficient)
      LeftAuthority13388.bound (LeftAuthority13388.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13389RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13388.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13388.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13411.bound LeftAuthority13388.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13411.bound, LeftAuthority13388.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13411.actual selector witness) * (LeftAuthority13388.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13415

namespace LeftBound13426
def owner : Owner := ⟨.program ⟨214⟩, ⟨15642⟩⟩
def transferEvent : Nat := 13426
def frameStart : Nat := 13338
def rule : BoundRule := .product (.predecessor 0 13424 .coefficient) (.predecessor 1 13425 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13424 .coefficient)
      LeftAuthority13399.bound (LeftAuthority13399.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13400RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13399.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13399.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13425 .coefficient)
      LeftAuthority13422.bound (LeftAuthority13422.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13423RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13422.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13422.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority13399.bound LeftAuthority13422.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13399.bound, LeftAuthority13422.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority13399.actual selector witness) * (LeftAuthority13422.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13426

namespace LeftBound13434
def owner : Owner := ⟨.program ⟨214⟩, ⟨15643⟩⟩
def transferEvent : Nat := 13434
def frameStart : Nat := 13338
def rule : BoundRule := .sum [.predecessor 0 13432 .coefficient, .predecessor 1 13433 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13432 .coefficient)
      LeftAuthority13430.bound (LeftAuthority13430.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13431RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13430.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13430.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13433 .coefficient)
      LeftBound13426.bound (LeftBound13426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13428RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13426.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13426.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority13430.bound, LeftBound13426.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13430.bound, LeftBound13426.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority13430.actual selector witness, LeftBound13426.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13434

namespace LeftBound13438
def owner : Owner := ⟨.program ⟨214⟩, ⟨27272⟩⟩
def transferEvent : Nat := 13438
def frameStart : Nat := 13338
def rule : BoundRule := .sum [.predecessor 0 13436 .coefficient, .predecessor 1 13437 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13436 .coefficient)
      LeftBound13434.bound (LeftBound13434.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13435RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13434.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13434.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13437 .coefficient)
      LeftBound13415.bound (LeftBound13415.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13415.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13415.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13434.bound, LeftBound13415.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13434.bound, LeftBound13415.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13434.actual selector witness, LeftBound13415.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13438

namespace LeftBound13451
def owner : Owner := ⟨.program ⟨214⟩, ⟨27270⟩⟩
def transferEvent : Nat := 13451
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13449 .coefficient, .predecessor 1 13450 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13449 .coefficient)
      LeftBound13280.bound (LeftBound13280.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13448RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13280.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13280.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13450 .coefficient)
      LeftBound13263.bound (LeftBound13263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13270RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13263.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13263.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13280.bound, LeftBound13263.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13280.bound, LeftBound13263.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13280.actual selector witness, LeftBound13263.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13451

namespace LeftBound13454
def owner : Owner := ⟨.program ⟨214⟩, ⟨27270⟩⟩
def transferEvent : Nat := 13454
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 13448 .summary, .result 13270 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13448 .summary)
      LeftBound13282.bound (LeftBound13282.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20987⟩⟩) (rawTerms := some (Proof.Events052.exact13448RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13282.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13270 .summary)
      LeftBound13265.bound (LeftBound13265.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27269⟩⟩) (rawTerms := some (Proof.Events051.exact13270RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13265.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13282.bound, LeftBound13265.bound]
def bound : CoeffClass := .finite ⟨1291978824159503986688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13282.bound, LeftBound13265.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13282.actual selector witness, LeftBound13265.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13454

namespace LeftBound13477
def owner : Owner := ⟨.program ⟨214⟩, ⟨89⟩⟩
def transferEvent : Nat := 13477
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 13476 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13476 .coefficient)
      LeftAuthority6440.bound (LeftAuthority6440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6440.derived selector witness)

def rawBound : CoeffClass := LeftAuthority6440.bound
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority6440.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound13477

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
