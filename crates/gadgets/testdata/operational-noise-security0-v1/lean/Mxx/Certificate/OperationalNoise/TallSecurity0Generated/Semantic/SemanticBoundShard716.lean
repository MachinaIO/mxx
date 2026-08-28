import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard657
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard715

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound104520
def owner : Owner := ⟨.program ⟨214⟩, ⟨29346⟩⟩
def transferEvent : Nat := 104520
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104514 .summary, .result 104360 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104514 .summary)
      LeftBound104372.bound (LeftBound104372.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22328⟩⟩) (rawTerms := some (Proof.Events408.exact104514RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104372.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104360 .summary)
      LeftBound104355.bound (LeftBound104355.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29345⟩⟩) (rawTerms := some (Proof.Events407.exact104360RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104355.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104372.bound, LeftBound104355.bound]
def bound : CoeffClass := .finite ⟨1292382248169874534400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104372.bound, LeftBound104355.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound104372.actual selector witness, LeftBound104355.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104520

namespace LeftBound104524
def owner : Owner := ⟨.program ⟨214⟩, ⟨29347⟩⟩
def transferEvent : Nat := 104524
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 104522 .coefficient) (.predecessor 1 104523 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104522 .coefficient)
      LeftBound104517.bound (LeftBound104517.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104517.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104517.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104523 .coefficient)
      LeftBound5578.bound (LeftBound5578.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5579RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5578.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5578.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound104517.bound LeftBound5578.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104517.bound, LeftBound5578.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound104517.actual selector witness) * (LeftBound5578.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104524

namespace LeftBound104525
def owner : Owner := ⟨.program ⟨214⟩, ⟨29347⟩⟩
def transferEvent : Nat := 104525
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6665⟩⟩]⟩ [⟨.result 5575 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5575 .coefficient)
      LeftAuthority5574.bound (LeftAuthority5574.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6665⟩⟩) (rawTerms := some (Proof.Events021.exact5575RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5574.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5574.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5574.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5574.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5574.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound104525

namespace LeftBound104526
def owner : Owner := ⟨.program ⟨214⟩, ⟨29347⟩⟩
def transferEvent : Nat := 104526
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 104521 .summary) (.transfer 104525) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104521 .summary)
      LeftBound104520.bound (LeftBound104520.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29346⟩⟩) (rawTerms := some (Proof.Events408.exact104521RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104520.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 104525)
      LeftBound104525.bound (LeftBound104525.actual selector witness) := by
  exact .transfer (LeftBound104525.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound104520.bound LeftBound104525.bound
def bound : CoeffClass := .finite ⟨4743063528899410259240550400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104520.bound, LeftBound104525.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound104520.actual selector witness) * (LeftBound104525.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104526

namespace LeftBound104541
def owner : Owner := ⟨.program ⟨214⟩, ⟨29128⟩⟩
def transferEvent : Nat := 104541
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 104539 .coefficient) (.predecessor 1 104540 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104539 .coefficient)
      LeftBound96356.bound (LeftBound96356.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96356.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96356.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104540 .coefficient)
      LeftAuthority104537.bound (LeftAuthority104537.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104537.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104537.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound96356.bound LeftAuthority104537.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96356.bound, LeftAuthority104537.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound96356.actual selector witness) * (LeftAuthority104537.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104541

namespace LeftBound104542
def owner : Owner := ⟨.program ⟨214⟩, ⟨29128⟩⟩
def transferEvent : Nat := 104542
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29126⟩⟩]⟩ [⟨.result 104538 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104538 .coefficient)
      LeftAuthority104537.bound (LeftAuthority104537.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29126⟩⟩) (rawTerms := some (Proof.Events408.exact104538RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104537.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104537.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority104537.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104537.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority104537.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound104542

namespace LeftBound104543
def owner : Owner := ⟨.program ⟨214⟩, ⟨29128⟩⟩
def transferEvent : Nat := 104543
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 96360 .summary) (.transfer 104542) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96360 .summary)
      LeftBound96359.bound (LeftBound96359.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25439⟩⟩) (rawTerms := some (Proof.Events376.exact96360RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96359.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 104542)
      LeftBound104542.bound (LeftBound104542.actual selector witness) := by
  exact .transfer (LeftBound104542.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound96359.bound LeftBound104542.bound
def bound : CoeffClass := .finite ⟨1292337421468529852416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96359.bound, LeftBound104542.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound96359.actual selector witness) * (LeftBound104542.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104543

namespace LeftBound104554
def owner : Owner := ⟨.program ⟨214⟩, ⟨22183⟩⟩
def transferEvent : Nat := 104554
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 104552 .coefficient) (.value (.predecessor 1 104553 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104552 .coefficient)
      LeftAuthority104550.bound (LeftAuthority104550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104551RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104550.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104550.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104553 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority104550.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104550.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority104550.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound104554

namespace LeftBound104558
def owner : Owner := ⟨.program ⟨214⟩, ⟨22184⟩⟩
def transferEvent : Nat := 104558
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 104556 .coefficient) (.predecessor 1 104557 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104556 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104557 .coefficient)
      LeftBound104554.bound (LeftBound104554.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104555RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104554.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104554.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound104554.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound104554.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound104554.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104558

namespace LeftBound104559
def owner : Owner := ⟨.program ⟨214⟩, ⟨22184⟩⟩
def transferEvent : Nat := 104559
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22181⟩⟩]⟩ [⟨.result 104551 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104551 .coefficient)
      LeftAuthority104550.bound (LeftAuthority104550.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22181⟩⟩) (rawTerms := some (Proof.Events408.exact104551RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104550.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104550.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority104550.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104550.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority104550.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound104559

namespace LeftBound104560
def owner : Owner := ⟨.program ⟨214⟩, ⟨22184⟩⟩
def transferEvent : Nat := 104560
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 104559) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 104559)
      LeftBound104559.bound (LeftBound104559.actual selector witness) := by
  exact .transfer (LeftBound104559.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound104559.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound104559.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound104559.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104560

namespace LeftBound104631
def owner : Owner := ⟨.program ⟨214⟩, ⟨16540⟩⟩
def transferEvent : Nat := 104631
def frameStart : Nat := 104604
def rule : BoundRule := .identity (.predecessor 0 104630 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104630 .coefficient)
      LeftAuthority104628.bound (LeftAuthority104628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104629RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104628.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104628.derived selector witness)

def rawBound : CoeffClass := LeftAuthority104628.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104628.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority104628.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound104631

namespace LeftBound104648
def owner : Owner := ⟨.program ⟨214⟩, ⟨16581⟩⟩
def transferEvent : Nat := 104648
def frameStart : Nat := 104604
def rule : BoundRule := .sum [.predecessor 0 104646 .coefficient, .predecessor 1 104647 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104646 .coefficient)
      LeftBound104631.bound (LeftBound104631.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound104631.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104647 .coefficient)
      LeftAuthority104644.bound (LeftAuthority104644.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority104644.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104631.bound, LeftAuthority104644.bound]
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104631.bound, LeftAuthority104644.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound104631.actual selector witness, LeftAuthority104644.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104648

namespace LeftBound104651
def owner : Owner := ⟨.program ⟨214⟩, ⟨16582⟩⟩
def transferEvent : Nat := 104651
def frameStart : Nat := 104604
def rule : BoundRule := .identity (.predecessor 0 104650 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104650 .coefficient)
      LeftBound104648.bound (LeftBound104648.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound104648.derived selector witness)

def rawBound : CoeffClass := LeftBound104648.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104648.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound104648.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound104651

namespace LeftBound104657
def owner : Owner := ⟨.program ⟨214⟩, ⟨16583⟩⟩
def transferEvent : Nat := 104657
def frameStart : Nat := 104604
def rule : BoundRule := .product (.predecessor 0 104655 .coefficient) (.predecessor 1 104656 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104655 .coefficient)
      LeftAuthority104653.bound (LeftAuthority104653.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104654RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104653.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104653.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104656 .coefficient)
      LeftBound104651.bound (LeftBound104651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104652RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104651.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104651.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority104653.bound LeftBound104651.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104653.bound, LeftBound104651.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority104653.actual selector witness) * (LeftBound104651.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104657

namespace LeftBound104665
def owner : Owner := ⟨.program ⟨214⟩, ⟨16584⟩⟩
def transferEvent : Nat := 104665
def frameStart : Nat := 104604
def rule : BoundRule := .sum [.predecessor 0 104663 .coefficient, .predecessor 1 104664 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104663 .coefficient)
      LeftAuthority104661.bound (LeftAuthority104661.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104662RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104661.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104661.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104664 .coefficient)
      LeftBound104657.bound (LeftBound104657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104657.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104657.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority104661.bound, LeftBound104657.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104661.bound, LeftBound104657.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority104661.actual selector witness, LeftBound104657.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104665

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
