import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard094
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard125

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound20470
def owner : Owner := ⟨.program ⟨214⟩, ⟨26829⟩⟩
def transferEvent : Nat := 20470
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 20464 .summary, .result 20286 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20464 .summary)
      LeftBound20298.bound (LeftBound20298.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20627⟩⟩) (rawTerms := some (Proof.Events079.exact20464RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20298.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20286 .summary)
      LeftBound20281.bound (LeftBound20281.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26828⟩⟩) (rawTerms := some (Proof.Events079.exact20286RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20281.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20298.bound, LeftBound20281.bound]
def bound : CoeffClass := .finite ⟨1291911586824442228736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20298.bound, LeftBound20281.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20298.actual selector witness, LeftBound20281.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20470

namespace LeftBound20474
def owner : Owner := ⟨.program ⟨214⟩, ⟨26830⟩⟩
def transferEvent : Nat := 20474
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 20472 .coefficient) (.predecessor 1 20473 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20472 .coefficient)
      LeftBound20467.bound (LeftBound20467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20471RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20467.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20473 .coefficient)
      LeftBound5818.bound (LeftBound5818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5819RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5818.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5818.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound20467.bound LeftBound5818.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20467.bound, LeftBound5818.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound20467.actual selector witness) * (LeftBound5818.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20474

namespace LeftBound20475
def owner : Owner := ⟨.program ⟨214⟩, ⟨26830⟩⟩
def transferEvent : Nat := 20475
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩ [⟨.result 5815 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5815 .coefficient)
      LeftAuthority5814.bound (LeftAuthority5814.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6663⟩⟩) (rawTerms := some (Proof.Events022.exact5815RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5814.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5814.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5814.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5814.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5814.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound20475

namespace LeftBound20476
def owner : Owner := ⟨.program ⟨214⟩, ⟨26830⟩⟩
def transferEvent : Nat := 20476
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 20471 .summary) (.transfer 20475) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20471 .summary)
      LeftBound20470.bound (LeftBound20470.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26829⟩⟩) (rawTerms := some (Proof.Events079.exact20471RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20470.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 20475)
      LeftBound20475.bound (LeftBound20475.actual selector witness) := by
  exact .transfer (LeftBound20475.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound20470.bound LeftBound20475.bound
def bound : CoeffClass := .finite ⟨4741336194231092170536779776, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20470.bound, LeftBound20475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound20470.actual selector witness) * (LeftBound20475.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20476

namespace LeftBound20491
def owner : Owner := ⟨.program ⟨214⟩, ⟨26611⟩⟩
def transferEvent : Nat := 20491
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 20489 .coefficient) (.predecessor 1 20490 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20489 .coefficient)
      LeftBound14759.bound (LeftBound14759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14763RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20490 .coefficient)
      LeftAuthority20487.bound (LeftAuthority20487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20487.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20487.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound14759.bound LeftAuthority20487.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14759.bound, LeftAuthority20487.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound14759.actual selector witness) * (LeftAuthority20487.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20491

namespace LeftBound20492
def owner : Owner := ⟨.program ⟨214⟩, ⟨26611⟩⟩
def transferEvent : Nat := 20492
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26609⟩⟩]⟩ [⟨.result 20488 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20488 .coefficient)
      LeftAuthority20487.bound (LeftAuthority20487.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26609⟩⟩) (rawTerms := some (Proof.Events080.exact20488RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20487.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20487.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority20487.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20487.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority20487.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound20492

namespace LeftBound20493
def owner : Owner := ⟨.program ⟨214⟩, ⟨26611⟩⟩
def transferEvent : Nat := 20493
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 14763 .summary) (.transfer 20492) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14763 .summary)
      LeftBound14762.bound (LeftBound14762.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25010⟩⟩) (rawTerms := some (Proof.Events057.exact14763RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound14762.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 20492)
      LeftBound20492.bound (LeftBound20492.actual selector witness) := by
  exact .transfer (LeftBound20492.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound14762.bound LeftBound20492.bound
def bound : CoeffClass := .finite ⟨1291900378790628425728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14762.bound, LeftBound20492.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound14762.actual selector witness) * (LeftBound20492.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20493

namespace LeftBound20504
def owner : Owner := ⟨.program ⟨214⟩, ⟨20482⟩⟩
def transferEvent : Nat := 20504
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 20502 .coefficient) (.value (.predecessor 1 20503 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20502 .coefficient)
      LeftAuthority20500.bound (LeftAuthority20500.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20501RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20500.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20500.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20503 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority20500.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20500.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority20500.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound20504

namespace LeftBound20508
def owner : Owner := ⟨.program ⟨214⟩, ⟨20483⟩⟩
def transferEvent : Nat := 20508
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 20506 .coefficient) (.predecessor 1 20507 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20506 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20507 .coefficient)
      LeftBound20504.bound (LeftBound20504.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20505RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20504.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20504.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound20504.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound20504.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound20504.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20508

namespace LeftBound20509
def owner : Owner := ⟨.program ⟨214⟩, ⟨20483⟩⟩
def transferEvent : Nat := 20509
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20480⟩⟩]⟩ [⟨.result 20501 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20501 .coefficient)
      LeftAuthority20500.bound (LeftAuthority20500.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20480⟩⟩) (rawTerms := some (Proof.Events080.exact20501RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20500.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20500.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority20500.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20500.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority20500.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound20509

namespace LeftBound20510
def owner : Owner := ⟨.program ⟨214⟩, ⟨20483⟩⟩
def transferEvent : Nat := 20510
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 20509) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 20509)
      LeftBound20509.bound (LeftBound20509.actual selector witness) := by
  exact .transfer (LeftBound20509.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound20509.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound20509.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound20509.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20510

namespace LeftBound20605
def owner : Owner := ⟨.program ⟨214⟩, ⟨14970⟩⟩
def transferEvent : Nat := 20605
def frameStart : Nat := 20566
def rule : BoundRule := .identity (.predecessor 0 20604 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20604 .coefficient)
      LeftAuthority20602.bound (LeftAuthority20602.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20603RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20602.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20602.derived selector witness)

def rawBound : CoeffClass := LeftAuthority20602.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20602.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority20602.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound20605

namespace LeftBound20622
def owner : Owner := ⟨.program ⟨214⟩, ⟨15009⟩⟩
def transferEvent : Nat := 20622
def frameStart : Nat := 20566
def rule : BoundRule := .sum [.predecessor 0 20620 .coefficient, .predecessor 1 20621 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20620 .coefficient)
      LeftBound20605.bound (LeftBound20605.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound20605.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20621 .coefficient)
      LeftAuthority20618.bound (LeftAuthority20618.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority20618.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20605.bound, LeftAuthority20618.bound]
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20605.bound, LeftAuthority20618.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20605.actual selector witness, LeftAuthority20618.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20622

namespace LeftBound20625
def owner : Owner := ⟨.program ⟨214⟩, ⟨15010⟩⟩
def transferEvent : Nat := 20625
def frameStart : Nat := 20566
def rule : BoundRule := .identity (.predecessor 0 20624 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20624 .coefficient)
      LeftBound20622.bound (LeftBound20622.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound20622.derived selector witness)

def rawBound : CoeffClass := LeftBound20622.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20622.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound20622.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound20625

namespace LeftBound20631
def owner : Owner := ⟨.program ⟨214⟩, ⟨15011⟩⟩
def transferEvent : Nat := 20631
def frameStart : Nat := 20566
def rule : BoundRule := .product (.predecessor 0 20629 .coefficient) (.predecessor 1 20630 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20629 .coefficient)
      LeftAuthority20627.bound (LeftAuthority20627.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20627.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20627.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20630 .coefficient)
      LeftBound20625.bound (LeftBound20625.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20626RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20625.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20625.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority20627.bound LeftBound20625.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20627.bound, LeftBound20625.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority20627.actual selector witness) * (LeftBound20625.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20631

namespace LeftBound20639
def owner : Owner := ⟨.program ⟨214⟩, ⟨15012⟩⟩
def transferEvent : Nat := 20639
def frameStart : Nat := 20566
def rule : BoundRule := .sum [.predecessor 0 20637 .coefficient, .predecessor 1 20638 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20637 .coefficient)
      LeftAuthority20635.bound (LeftAuthority20635.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20635.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20635.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20638 .coefficient)
      LeftBound20631.bound (LeftBound20631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20633RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20631.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20631.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority20635.bound, LeftBound20631.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20635.bound, LeftBound20631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority20635.actual selector witness, LeftBound20631.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20639

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
