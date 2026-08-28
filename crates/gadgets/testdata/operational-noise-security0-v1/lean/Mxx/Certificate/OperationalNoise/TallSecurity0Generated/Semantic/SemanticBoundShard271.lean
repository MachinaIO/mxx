import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard270

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound40455
def owner : Owner := ⟨.program ⟨214⟩, ⟨14450⟩⟩
def transferEvent : Nat := 40455
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40453 .coefficient, .predecessor 1 40454 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40453 .coefficient)
      LeftBound40445.bound (LeftBound40445.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40452RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40445.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40445.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40454 .coefficient)
      LeftBound40417.bound (LeftBound40417.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40422RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40417.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40417.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40445.bound, LeftBound40417.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40445.bound, LeftBound40417.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40445.actual selector witness, LeftBound40417.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40455

namespace LeftBound40457
def owner : Owner := ⟨.program ⟨214⟩, ⟨14450⟩⟩
def transferEvent : Nat := 40457
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 40452 .summary, .result 40422 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40452 .summary)
      LeftBound40447.bound (LeftBound40447.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14449⟩⟩) (rawTerms := some (Proof.Events158.exact40452RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40447.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40422 .summary)
      LeftBound40419.bound (LeftBound40419.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14445⟩⟩) (rawTerms := some (Proof.Events157.exact40422RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40419.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40447.bound, LeftBound40419.bound]
def bound : CoeffClass := .finite ⟨95438720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40447.bound, LeftBound40419.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40447.actual selector witness, LeftBound40419.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40457

namespace LeftBound40461
def owner : Owner := ⟨.program ⟨214⟩, ⟨26154⟩⟩
def transferEvent : Nat := 40461
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 40459 .coefficient) (.predecessor 1 40460 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40459 .coefficient)
      LeftBound40455.bound (LeftBound40455.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40458RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40455.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40455.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40460 .coefficient)
      LeftAuthority40393.bound (LeftAuthority40393.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40394RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40393.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40393.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40455.bound LeftAuthority40393.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40455.bound, LeftAuthority40393.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40455.actual selector witness) * (LeftAuthority40393.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40461

namespace LeftBound40462
def owner : Owner := ⟨.program ⟨214⟩, ⟨26154⟩⟩
def transferEvent : Nat := 40462
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26153⟩⟩]⟩ [⟨.result 40394 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40394 .coefficient)
      LeftAuthority40393.bound (LeftAuthority40393.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26153⟩⟩) (rawTerms := some (Proof.Events157.exact40394RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40393.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40393.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority40393.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40393.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority40393.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound40462

namespace LeftBound40463
def owner : Owner := ⟨.program ⟨214⟩, ⟨26154⟩⟩
def transferEvent : Nat := 40463
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 40458 .summary) (.transfer 40462) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40458 .summary)
      LeftBound40457.bound (LeftBound40457.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14450⟩⟩) (rawTerms := some (Proof.Events158.exact40458RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40457.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 40462)
      LeftBound40462.bound (LeftBound40462.actual selector witness) := by
  exact .transfer (LeftBound40462.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40457.bound LeftBound40462.bound
def bound : CoeffClass := .finite ⟨350261629419520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40457.bound, LeftBound40462.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40457.actual selector witness) * (LeftBound40462.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40463

namespace LeftBound40474
def owner : Owner := ⟨.program ⟨214⟩, ⟨19610⟩⟩
def transferEvent : Nat := 40474
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 40472 .coefficient) (.value (.predecessor 1 40473 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40472 .coefficient)
      LeftAuthority40470.bound (LeftAuthority40470.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40471RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40470.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40470.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40473 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority40470.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40470.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority40470.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound40474

namespace LeftBound40478
def owner : Owner := ⟨.program ⟨214⟩, ⟨19611⟩⟩
def transferEvent : Nat := 40478
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 40476 .coefficient) (.predecessor 1 40477 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40476 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40477 .coefficient)
      LeftBound40474.bound (LeftBound40474.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40474.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40474.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound40474.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound40474.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound40474.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40478

namespace LeftBound40479
def owner : Owner := ⟨.program ⟨214⟩, ⟨19611⟩⟩
def transferEvent : Nat := 40479
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19608⟩⟩]⟩ [⟨.result 40471 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40471 .coefficient)
      LeftAuthority40470.bound (LeftAuthority40470.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19608⟩⟩) (rawTerms := some (Proof.Events158.exact40471RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40470.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40470.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority40470.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40470.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority40470.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound40479

namespace LeftBound40480
def owner : Owner := ⟨.program ⟨214⟩, ⟨19611⟩⟩
def transferEvent : Nat := 40480
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 40479) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 40479)
      LeftBound40479.bound (LeftBound40479.actual selector witness) := by
  exact .transfer (LeftBound40479.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound40479.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound40479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound40479.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40480

namespace LeftBound40559
def owner : Owner := ⟨.program ⟨214⟩, ⟨14443⟩⟩
def transferEvent : Nat := 40559
def frameStart : Nat := 40530
def rule : BoundRule := .product (.predecessor 0 40557 .coefficient) (.predecessor 1 40558 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40557 .coefficient)
      LeftAuthority40555.bound (LeftAuthority40555.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40556RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40555.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40555.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40558 .coefficient)
      LeftAuthority40552.bound (LeftAuthority40552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40552.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40552.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority40555.bound LeftAuthority40552.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40555.bound, LeftAuthority40552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority40555.actual selector witness) * (LeftAuthority40552.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40559

namespace LeftBound40563
def owner : Owner := ⟨.program ⟨214⟩, ⟨14444⟩⟩
def transferEvent : Nat := 40563
def frameStart : Nat := 40530
def rule : BoundRule := .identity (.predecessor 0 40562 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40562 .coefficient)
      LeftBound40559.bound (LeftBound40559.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40559.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40559.derived selector witness)

def rawBound : CoeffClass := LeftBound40559.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40559.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound40559.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound40563

namespace LeftBound40580
def owner : Owner := ⟨.program ⟨214⟩, ⟨14539⟩⟩
def transferEvent : Nat := 40580
def frameStart : Nat := 40530
def rule : BoundRule := .sum [.predecessor 0 40578 .coefficient, .predecessor 1 40579 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40578 .coefficient)
      LeftBound40563.bound (LeftBound40563.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound40563.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40579 .coefficient)
      LeftAuthority40576.bound (LeftAuthority40576.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority40576.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40563.bound, LeftAuthority40576.bound]
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40563.bound, LeftAuthority40576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40563.actual selector witness, LeftAuthority40576.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40580

namespace LeftBound40583
def owner : Owner := ⟨.program ⟨214⟩, ⟨14540⟩⟩
def transferEvent : Nat := 40583
def frameStart : Nat := 40530
def rule : BoundRule := .identity (.predecessor 0 40582 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40582 .coefficient)
      LeftBound40580.bound (LeftBound40580.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound40580.derived selector witness)

def rawBound : CoeffClass := LeftBound40580.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40580.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound40580.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound40583

namespace LeftBound40589
def owner : Owner := ⟨.program ⟨214⟩, ⟨14541⟩⟩
def transferEvent : Nat := 40589
def frameStart : Nat := 40530
def rule : BoundRule := .product (.predecessor 0 40587 .coefficient) (.predecessor 1 40588 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40587 .coefficient)
      LeftAuthority40585.bound (LeftAuthority40585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40585.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40585.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40588 .coefficient)
      LeftBound40583.bound (LeftBound40583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40583.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40583.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority40585.bound LeftBound40583.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40585.bound, LeftBound40583.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority40585.actual selector witness) * (LeftBound40583.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40589

namespace LeftBound40605
def owner : Owner := ⟨.program ⟨214⟩, ⟨7856⟩⟩
def transferEvent : Nat := 40605
def frameStart : Nat := 40530
def rule : BoundRule := .scale (.predecessor 0 40603 .coefficient) (.value (.predecessor 1 40604 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40603 .coefficient)
      LeftAuthority40601.bound (LeftAuthority40601.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40602RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40601.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40601.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40604 .coefficient)
      LeftAuthority40592.bound (LeftAuthority40592.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority40592.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority40601.bound LeftAuthority40592.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40601.bound, LeftAuthority40592.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority40601.actual selector witness) * (LeftAuthority40592.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound40605

namespace LeftBound40608
def owner : Owner := ⟨.program ⟨214⟩, ⟨6761⟩⟩
def transferEvent : Nat := 40608
def frameStart : Nat := 40530
def rule : BoundRule := .identity (.predecessor 0 40607 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40607 .coefficient)
      LeftAuthority40595.bound (LeftAuthority40595.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40596RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40595.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40595.derived selector witness)

def rawBound : CoeffClass := LeftAuthority40595.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40595.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority40595.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound40608

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
