import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard028
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard029
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound21438
def owner : Owner := ⟨.program ⟨214⟩, ⟨13379⟩⟩
def transferEvent : Nat := 21438
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨104⟩⟩]⟩ [⟨.result 6444 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6444 .coefficient)
      LeftBound6443.bound (LeftBound6443.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨104⟩⟩) (rawTerms := some (Proof.Events025.exact6444RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6443.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6443.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound6443.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound6443.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound21438

namespace LeftBound21443
def owner : Owner := ⟨.program ⟨214⟩, ⟨13380⟩⟩
def transferEvent : Nat := 21443
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21441 .coefficient) (.predecessor 1 21442 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21441 .coefficient)
      LeftBound21437.bound (LeftBound21437.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21440RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21437.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21437.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21442 .coefficient)
      LeftAuthority844.bound (LeftAuthority844.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority844.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority844.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound21437.bound LeftAuthority844.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21437.bound, LeftAuthority844.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound21437.actual selector witness) * (LeftAuthority844.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21443

namespace LeftBound21444
def owner : Owner := ⟨.program ⟨214⟩, ⟨13380⟩⟩
def transferEvent : Nat := 21444
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨10360⟩⟩], []⟩ [⟨.result 845 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 845 .coefficient)
      LeftAuthority844.bound (LeftAuthority844.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨10360⟩⟩) (rawTerms := some (Proof.Events003.exact845RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority844.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority844.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority844.bound []
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority844.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority844.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound21444

namespace LeftBound21445
def owner : Owner := ⟨.program ⟨214⟩, ⟨13380⟩⟩
def transferEvent : Nat := 21445
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21440 .summary) (.transfer 21444) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21440 .summary)
      LeftBound21438.bound (LeftBound21438.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13379⟩⟩) (rawTerms := some (Proof.Events083.exact21440RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21438.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 21444)
      LeftBound21444.bound (LeftBound21444.actual selector witness) := by
  exact .transfer (LeftBound21444.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21438.bound LeftBound21444.bound
def bound : CoeffClass := .finite ⟨49920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21438.bound, LeftBound21444.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21438.actual selector witness) * (LeftBound21444.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21445

namespace LeftBound21451
def owner : Owner := ⟨.program ⟨214⟩, ⟨10361⟩⟩
def transferEvent : Nat := 21451
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 21449 .coefficient) (.predecessor 1 21450 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21449 .coefficient)
      LeftAuthority844.bound (LeftAuthority844.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority844.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority844.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21450 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority844.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority844.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority844.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound21451

namespace LeftBound21456
def owner : Owner := ⟨.program ⟨214⟩, ⟨7340⟩⟩
def transferEvent : Nat := 21456
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21454 .coefficient) (.predecessor 1 21455 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21454 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21455 .coefficient)
      LeftBound6497.bound (LeftBound6497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6497.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6497.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound6497.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound6497.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound6497.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21456

namespace LeftBound21461
def owner : Owner := ⟨.program ⟨214⟩, ⟨10362⟩⟩
def transferEvent : Nat := 21461
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21459 .coefficient, .predecessor 1 21460 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21459 .coefficient)
      LeftBound21456.bound (LeftBound21456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21458RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21456.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21456.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21460 .coefficient)
      LeftBound21451.bound (LeftBound21451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21453RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21451.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21451.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21456.bound, LeftBound21451.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21456.bound, LeftBound21451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21456.actual selector witness, LeftBound21451.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21461

namespace LeftBound21465
def owner : Owner := ⟨.program ⟨214⟩, ⟨10363⟩⟩
def transferEvent : Nat := 21465
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21463 .coefficient, .predecessor 1 21464 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21463 .coefficient)
      LeftBound21461.bound (LeftBound21461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21461.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21461.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21464 .coefficient)
      LeftBound6489.bound (LeftBound6489.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6489.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6489.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21461.bound, LeftBound6489.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21461.bound, LeftBound6489.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21461.actual selector witness, LeftBound6489.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21465

namespace LeftBound21466
def owner : Owner := ⟨.program ⟨214⟩, ⟨10363⟩⟩
def transferEvent : Nat := 21466
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨84⟩⟩]⟩ [⟨.result 6490 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6490 .coefficient)
      LeftBound6489.bound (LeftBound6489.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨84⟩⟩) (rawTerms := some (Proof.Events025.exact6490RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6489.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6489.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound6489.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6489.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound6489.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound21466

namespace LeftBound21471
def owner : Owner := ⟨.program ⟨214⟩, ⟨10364⟩⟩
def transferEvent : Nat := 21471
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21469 .coefficient) (.predecessor 1 21470 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21469 .coefficient)
      LeftBound21465.bound (LeftBound21465.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21465.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21470 .coefficient)
      LeftBound6486.bound (LeftBound6486.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6487RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6486.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6486.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound21465.bound LeftBound6486.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21465.bound, LeftBound6486.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound21465.actual selector witness) * (LeftBound6486.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21471

namespace LeftBound21472
def owner : Owner := ⟨.program ⟨214⟩, ⟨10364⟩⟩
def transferEvent : Nat := 21472
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩ [⟨.result 6483 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6483 .coefficient)
      LeftAuthority6482.bound (LeftAuthority6482.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7882⟩⟩) (rawTerms := some (Proof.Events025.exact6483RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6482.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6482.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6482.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6482.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6482.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound21472

namespace LeftBound21473
def owner : Owner := ⟨.program ⟨214⟩, ⟨10364⟩⟩
def transferEvent : Nat := 21473
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21468 .summary) (.transfer 21472) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21468 .summary)
      LeftBound21466.bound (LeftBound21466.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10363⟩⟩) (rawTerms := some (Proof.Events083.exact21468RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21466.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 21472)
      LeftBound21472.bound (LeftBound21472.actual selector witness) := by
  exact .transfer (LeftBound21472.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound21466.bound LeftBound21472.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21466.bound, LeftBound21472.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound21466.actual selector witness) * (LeftBound21472.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21473

namespace LeftBound21481
def owner : Owner := ⟨.program ⟨214⟩, ⟨13381⟩⟩
def transferEvent : Nat := 21481
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21479 .coefficient, .predecessor 1 21480 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21479 .coefficient)
      LeftBound21471.bound (LeftBound21471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21471.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21480 .coefficient)
      LeftBound21443.bound (LeftBound21443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21448RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21443.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21443.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21471.bound, LeftBound21443.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21471.bound, LeftBound21443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21471.actual selector witness, LeftBound21443.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21481

namespace LeftBound21483
def owner : Owner := ⟨.program ⟨214⟩, ⟨13381⟩⟩
def transferEvent : Nat := 21483
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 21478 .summary, .result 21448 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21478 .summary)
      LeftBound21473.bound (LeftBound21473.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10364⟩⟩) (rawTerms := some (Proof.Events083.exact21478RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21473.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21448 .summary)
      LeftBound21445.bound (LeftBound21445.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13380⟩⟩) (rawTerms := some (Proof.Events083.exact21448RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21445.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21473.bound, LeftBound21445.bound]
def bound : CoeffClass := .finite ⟨95470336, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21473.bound, LeftBound21445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21473.actual selector witness, LeftBound21445.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21483

namespace LeftBound21487
def owner : Owner := ⟨.program ⟨214⟩, ⟨25774⟩⟩
def transferEvent : Nat := 21487
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21485 .coefficient) (.predecessor 1 21486 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 21485 .coefficient)
      LeftBound21481.bound (LeftBound21481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21481.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21481.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 21486 .coefficient)
      LeftAuthority21414.bound (LeftAuthority21414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21414.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21414.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound21481.bound LeftAuthority21414.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21481.bound, LeftAuthority21414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound21481.actual selector witness) * (LeftAuthority21414.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21487

namespace LeftBound21488
def owner : Owner := ⟨.program ⟨214⟩, ⟨25774⟩⟩
def transferEvent : Nat := 21488
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25773⟩⟩]⟩ [⟨.result 21415 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21415 .coefficient)
      LeftAuthority21414.bound (LeftAuthority21414.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25773⟩⟩) (rawTerms := some (Proof.Events083.exact21415RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21414.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21414.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority21414.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority21414.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound21488

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
