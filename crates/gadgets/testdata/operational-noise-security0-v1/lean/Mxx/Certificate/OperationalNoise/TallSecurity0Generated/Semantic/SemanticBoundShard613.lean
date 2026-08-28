import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard547
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard612

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound90322
def owner : Owner := ⟨.program ⟨214⟩, ⟨17051⟩⟩
def transferEvent : Nat := 90322
def frameStart : Nat := 90266
def rule : BoundRule := .sum [.predecessor 0 90320 .coefficient, .predecessor 1 90321 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90320 .coefficient)
      LeftBound90305.bound (LeftBound90305.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound90305.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90321 .coefficient)
      LeftAuthority90318.bound (LeftAuthority90318.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority90318.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90305.bound, LeftAuthority90318.bound]
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90305.bound, LeftAuthority90318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound90305.actual selector witness, LeftAuthority90318.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90322

namespace LeftBound90325
def owner : Owner := ⟨.program ⟨214⟩, ⟨17052⟩⟩
def transferEvent : Nat := 90325
def frameStart : Nat := 90266
def rule : BoundRule := .identity (.predecessor 0 90324 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90324 .coefficient)
      LeftBound90322.bound (LeftBound90322.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound90322.derived selector witness)

def rawBound : CoeffClass := LeftBound90322.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90322.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound90322.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound90325

namespace LeftBound90331
def owner : Owner := ⟨.program ⟨214⟩, ⟨17053⟩⟩
def transferEvent : Nat := 90331
def frameStart : Nat := 90266
def rule : BoundRule := .product (.predecessor 0 90329 .coefficient) (.predecessor 1 90330 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90329 .coefficient)
      LeftAuthority90327.bound (LeftAuthority90327.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events352.exact90328RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90327.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90327.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90330 .coefficient)
      LeftBound90325.bound (LeftBound90325.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events352.exact90326RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90325.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90325.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority90327.bound LeftBound90325.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90327.bound, LeftBound90325.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority90327.actual selector witness) * (LeftBound90325.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90331

namespace LeftBound90339
def owner : Owner := ⟨.program ⟨214⟩, ⟨17054⟩⟩
def transferEvent : Nat := 90339
def frameStart : Nat := 90266
def rule : BoundRule := .sum [.predecessor 0 90337 .coefficient, .predecessor 1 90338 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90337 .coefficient)
      LeftAuthority90335.bound (LeftAuthority90335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events352.exact90336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90335.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90335.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90338 .coefficient)
      LeftBound90331.bound (LeftBound90331.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events352.exact90333RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90331.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90331.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority90335.bound, LeftBound90331.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90335.bound, LeftBound90331.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority90335.actual selector witness, LeftBound90331.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90339

namespace LeftBound90343
def owner : Owner := ⟨.program ⟨214⟩, ⟨30110⟩⟩
def transferEvent : Nat := 90343
def frameStart : Nat := 90266
def rule : BoundRule := .product (.predecessor 0 90341 .coefficient) (.predecessor 1 90342 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90341 .coefficient)
      LeftBound90339.bound (LeftBound90339.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events352.exact90340RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90339.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90339.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90342 .coefficient)
      LeftAuthority90316.bound (LeftAuthority90316.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events352.exact90317RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90316.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90316.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound90339.bound LeftAuthority90316.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90339.bound, LeftAuthority90316.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound90339.actual selector witness) * (LeftAuthority90316.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90343

namespace LeftBound90354
def owner : Owner := ⟨.program ⟨214⟩, ⟨18126⟩⟩
def transferEvent : Nat := 90354
def frameStart : Nat := 90266
def rule : BoundRule := .product (.predecessor 0 90352 .coefficient) (.predecessor 1 90353 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90352 .coefficient)
      LeftAuthority90327.bound (LeftAuthority90327.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events352.exact90328RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90327.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90327.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90353 .coefficient)
      LeftAuthority90350.bound (LeftAuthority90350.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events352.exact90351RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90350.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90350.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority90327.bound LeftAuthority90350.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90327.bound, LeftAuthority90350.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority90327.actual selector witness) * (LeftAuthority90350.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90354

namespace LeftBound90362
def owner : Owner := ⟨.program ⟨214⟩, ⟨18127⟩⟩
def transferEvent : Nat := 90362
def frameStart : Nat := 90266
def rule : BoundRule := .sum [.predecessor 0 90360 .coefficient, .predecessor 1 90361 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90360 .coefficient)
      LeftAuthority90358.bound (LeftAuthority90358.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events352.exact90359RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90358.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90358.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90361 .coefficient)
      LeftBound90354.bound (LeftBound90354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events352.exact90356RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90354.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority90358.bound, LeftBound90354.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90358.bound, LeftBound90354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority90358.actual selector witness, LeftBound90354.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90362

namespace LeftBound90366
def owner : Owner := ⟨.program ⟨214⟩, ⟨30115⟩⟩
def transferEvent : Nat := 90366
def frameStart : Nat := 90266
def rule : BoundRule := .sum [.predecessor 0 90364 .coefficient, .predecessor 1 90365 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90364 .coefficient)
      LeftBound90362.bound (LeftBound90362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events352.exact90363RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90362.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90362.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90365 .coefficient)
      LeftBound90343.bound (LeftBound90343.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events352.exact90348RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90343.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90343.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90362.bound, LeftBound90343.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90362.bound, LeftBound90343.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound90362.actual selector witness, LeftBound90343.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90366

namespace LeftBound90379
def owner : Owner := ⟨.program ⟨214⟩, ⟨30112⟩⟩
def transferEvent : Nat := 90379
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 90377 .coefficient, .predecessor 1 90378 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90377 .coefficient)
      LeftBound90208.bound (LeftBound90208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90376RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90208.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90208.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90378 .coefficient)
      LeftBound90191.bound (LeftBound90191.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events352.exact90198RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90191.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90191.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90208.bound, LeftBound90191.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90208.bound, LeftBound90191.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound90208.actual selector witness, LeftBound90191.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90379

namespace LeftBound90382
def owner : Owner := ⟨.program ⟨214⟩, ⟨30112⟩⟩
def transferEvent : Nat := 90382
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 90376 .summary, .result 90198 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 90376 .summary)
      LeftBound90210.bound (LeftBound90210.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22771⟩⟩) (rawTerms := some (Proof.Events353.exact90376RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90210.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 90198 .summary)
      LeftBound90193.bound (LeftBound90193.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30111⟩⟩) (rawTerms := some (Proof.Events352.exact90198RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90193.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90210.bound, LeftBound90193.bound]
def bound : CoeffClass := .finite ⟨1292539135285018636288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90210.bound, LeftBound90193.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound90210.actual selector witness, LeftBound90193.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90382

namespace LeftBound90386
def owner : Owner := ⟨.program ⟨214⟩, ⟨30113⟩⟩
def transferEvent : Nat := 90386
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 90384 .coefficient) (.predecessor 1 90385 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90384 .coefficient)
      LeftBound90379.bound (LeftBound90379.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90383RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90379.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90379.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90385 .coefficient)
      LeftBound5518.bound (LeftBound5518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5519RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5518.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound90379.bound LeftBound5518.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90379.bound, LeftBound5518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound90379.actual selector witness) * (LeftBound5518.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90386

namespace LeftBound90387
def owner : Owner := ⟨.program ⟨214⟩, ⟨30113⟩⟩
def transferEvent : Nat := 90387
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6657⟩⟩]⟩ [⟨.result 5515 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5515 .coefficient)
      LeftAuthority5514.bound (LeftAuthority5514.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6657⟩⟩) (rawTerms := some (Proof.Events021.exact5515RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5514.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5514.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5514.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5514.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound90387

namespace LeftBound90388
def owner : Owner := ⟨.program ⟨214⟩, ⟨30113⟩⟩
def transferEvent : Nat := 90388
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 90383 .summary) (.transfer 90387) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 90383 .summary)
      LeftBound90382.bound (LeftBound90382.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30112⟩⟩) (rawTerms := some (Proof.Events353.exact90383RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90382.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 90387)
      LeftBound90387.bound (LeftBound90387.actual selector witness) := by
  exact .transfer (LeftBound90387.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound90382.bound LeftBound90387.bound
def bound : CoeffClass := .finite ⟨4743639307122182955475140608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90382.bound, LeftBound90387.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound90382.actual selector witness) * (LeftBound90387.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90388

namespace LeftBound90403
def owner : Owner := ⟨.program ⟨214⟩, ⟨29814⟩⟩
def transferEvent : Nat := 90403
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 90401 .coefficient) (.predecessor 1 90402 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90401 .coefficient)
      LeftBound80672.bound (LeftBound80672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events315.exact80676RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80672.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80672.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90402 .coefficient)
      LeftAuthority90399.bound (LeftAuthority90399.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90400RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90399.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90399.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound80672.bound LeftAuthority90399.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80672.bound, LeftAuthority90399.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound80672.actual selector witness) * (LeftAuthority90399.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90403

namespace LeftBound90404
def owner : Owner := ⟨.program ⟨214⟩, ⟨29814⟩⟩
def transferEvent : Nat := 90404
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29812⟩⟩]⟩ [⟨.result 90400 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 90400 .coefficient)
      LeftAuthority90399.bound (LeftAuthority90399.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29812⟩⟩) (rawTerms := some (Proof.Events353.exact90400RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90399.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90399.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority90399.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90399.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority90399.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound90404

namespace LeftBound90405
def owner : Owner := ⟨.program ⟨214⟩, ⟨29814⟩⟩
def transferEvent : Nat := 90405
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80676 .summary) (.transfer 90404) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80676 .summary)
      LeftBound80675.bound (LeftBound80675.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25683⟩⟩) (rawTerms := some (Proof.Events315.exact80676RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80675.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 90404)
      LeftBound90404.bound (LeftBound90404.actual selector witness) := by
  exact .transfer (LeftBound90404.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound80675.bound LeftBound90404.bound
def bound : CoeffClass := .finite ⟨1292516721028694540288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80675.bound, LeftBound90404.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound80675.actual selector witness) * (LeftBound90404.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90405

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
