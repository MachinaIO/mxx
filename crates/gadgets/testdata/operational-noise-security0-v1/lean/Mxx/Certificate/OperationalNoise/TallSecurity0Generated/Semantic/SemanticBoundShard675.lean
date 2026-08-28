import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard673
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard674

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound98501
def owner : Owner := ⟨.program ⟨214⟩, ⟨16051⟩⟩
def transferEvent : Nat := 98501
def frameStart : Nat := 98411
def rule : BoundRule := .product (.predecessor 0 98499 .coefficient) (.predecessor 1 98500 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98499 .coefficient)
      LeftAuthority98454.bound (LeftAuthority98454.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98455RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98454.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98454.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98500 .coefficient)
      LeftAuthority98497.bound (LeftAuthority98497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98497.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98497.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority98454.bound LeftAuthority98497.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98454.bound, LeftAuthority98497.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority98454.actual selector witness) * (LeftAuthority98497.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98501

namespace LeftBound98509
def owner : Owner := ⟨.program ⟨214⟩, ⟨16052⟩⟩
def transferEvent : Nat := 98509
def frameStart : Nat := 98411
def rule : BoundRule := .sum [.predecessor 0 98507 .coefficient, .predecessor 1 98508 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98507 .coefficient)
      LeftAuthority98505.bound (LeftAuthority98505.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98506RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98505.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98505.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98508 .coefficient)
      LeftBound98501.bound (LeftBound98501.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98501.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98501.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority98505.bound, LeftBound98501.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98505.bound, LeftBound98501.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority98505.actual selector witness, LeftBound98501.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98509

namespace LeftBound98513
def owner : Owner := ⟨.program ⟨214⟩, ⟨26134⟩⟩
def transferEvent : Nat := 98513
def frameStart : Nat := 98411
def rule : BoundRule := .sum [.predecessor 0 98511 .coefficient, .predecessor 1 98512 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98511 .coefficient)
      LeftBound98509.bound (LeftBound98509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98512 .coefficient)
      LeftBound98490.bound (LeftBound98490.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98490.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98490.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98509.bound, LeftBound98490.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98509.bound, LeftBound98490.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98509.actual selector witness, LeftBound98490.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98513

namespace LeftBound98526
def owner : Owner := ⟨.program ⟨214⟩, ⟨26132⟩⟩
def transferEvent : Nat := 98526
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 98524 .coefficient, .predecessor 1 98525 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98524 .coefficient)
      LeftBound98371.bound (LeftBound98371.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98371.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98371.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98525 .coefficient)
      LeftBound98354.bound (LeftBound98354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98354.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98371.bound, LeftBound98354.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98371.bound, LeftBound98354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98371.actual selector witness, LeftBound98354.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98526

namespace LeftBound98529
def owner : Owner := ⟨.program ⟨214⟩, ⟨26132⟩⟩
def transferEvent : Nat := 98529
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 98523 .summary, .result 98361 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98523 .summary)
      LeftBound98373.bound (LeftBound98373.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19592⟩⟩) (rawTerms := some (Proof.Events384.exact98523RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98373.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98361 .summary)
      LeftBound98356.bound (LeftBound98356.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26131⟩⟩) (rawTerms := some (Proof.Events384.exact98361RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98356.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98373.bound, LeftBound98356.bound]
def bound : CoeffClass := .finite ⟨352072932929536, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98373.bound, LeftBound98356.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98373.actual selector witness, LeftBound98356.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98529

namespace LeftBound98533
def owner : Owner := ⟨.program ⟨214⟩, ⟨28050⟩⟩
def transferEvent : Nat := 98533
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 98531 .coefficient) (.predecessor 1 98532 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98531 .coefficient)
      LeftBound98526.bound (LeftBound98526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98526.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98532 .coefficient)
      LeftAuthority98276.bound (LeftAuthority98276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events383.exact98277RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98276.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98276.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98526.bound LeftAuthority98276.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98526.bound, LeftAuthority98276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98526.actual selector witness) * (LeftAuthority98276.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98533

namespace LeftBound98534
def owner : Owner := ⟨.program ⟨214⟩, ⟨28050⟩⟩
def transferEvent : Nat := 98534
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28048⟩⟩]⟩ [⟨.result 98277 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98277 .coefficient)
      LeftAuthority98276.bound (LeftAuthority98276.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28048⟩⟩) (rawTerms := some (Proof.Events383.exact98277RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98276.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98276.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority98276.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority98276.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound98534

namespace LeftBound98535
def owner : Owner := ⟨.program ⟨214⟩, ⟨28050⟩⟩
def transferEvent : Nat := 98535
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 98530 .summary) (.transfer 98534) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98530 .summary)
      LeftBound98529.bound (LeftBound98529.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26132⟩⟩) (rawTerms := some (Proof.Events384.exact98530RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98529.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 98534)
      LeftBound98534.bound (LeftBound98534.actual selector witness) := by
  exact .transfer (LeftBound98534.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98529.bound LeftBound98534.bound
def bound : CoeffClass := .finite ⟨1292113297018323992576, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98529.bound, LeftBound98534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98529.actual selector witness) * (LeftBound98534.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98535

namespace LeftBound98546
def owner : Owner := ⟨.program ⟨214⟩, ⟨21535⟩⟩
def transferEvent : Nat := 98546
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 98544 .coefficient) (.value (.predecessor 1 98545 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98544 .coefficient)
      LeftAuthority98542.bound (LeftAuthority98542.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98542.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98542.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98545 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority98542.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98542.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority98542.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound98546

namespace LeftBound98550
def owner : Owner := ⟨.program ⟨214⟩, ⟨21536⟩⟩
def transferEvent : Nat := 98550
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 98548 .coefficient) (.predecessor 1 98549 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98548 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98549 .coefficient)
      LeftBound98546.bound (LeftBound98546.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98547RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98546.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98546.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound98546.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound98546.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound98546.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98550

namespace LeftBound98551
def owner : Owner := ⟨.program ⟨214⟩, ⟨21536⟩⟩
def transferEvent : Nat := 98551
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21533⟩⟩]⟩ [⟨.result 98543 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98543 .coefficient)
      LeftAuthority98542.bound (LeftAuthority98542.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21533⟩⟩) (rawTerms := some (Proof.Events384.exact98543RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98542.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98542.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority98542.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98542.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority98542.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound98551

namespace LeftBound98552
def owner : Owner := ⟨.program ⟨214⟩, ⟨21536⟩⟩
def transferEvent : Nat := 98552
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 98551) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 98551)
      LeftBound98551.bound (LeftBound98551.actual selector witness) := by
  exact .transfer (LeftBound98551.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound98551.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound98551.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound98551.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98552

namespace LeftBound98623
def owner : Owner := ⟨.program ⟨214⟩, ⟨16050⟩⟩
def transferEvent : Nat := 98623
def frameStart : Nat := 98596
def rule : BoundRule := .identity (.predecessor 0 98622 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98622 .coefficient)
      LeftAuthority98620.bound (LeftAuthority98620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98621RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98620.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98620.derived selector witness)

def rawBound : CoeffClass := LeftAuthority98620.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98620.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority98620.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound98623

namespace LeftBound98640
def owner : Owner := ⟨.program ⟨214⟩, ⟨16126⟩⟩
def transferEvent : Nat := 98640
def frameStart : Nat := 98596
def rule : BoundRule := .sum [.predecessor 0 98638 .coefficient, .predecessor 1 98639 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98638 .coefficient)
      LeftBound98623.bound (LeftBound98623.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound98623.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98639 .coefficient)
      LeftAuthority98636.bound (LeftAuthority98636.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority98636.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98623.bound, LeftAuthority98636.bound]
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98623.bound, LeftAuthority98636.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98623.actual selector witness, LeftAuthority98636.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98640

namespace LeftBound98643
def owner : Owner := ⟨.program ⟨214⟩, ⟨16127⟩⟩
def transferEvent : Nat := 98643
def frameStart : Nat := 98596
def rule : BoundRule := .identity (.predecessor 0 98642 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98642 .coefficient)
      LeftBound98640.bound (LeftBound98640.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound98640.derived selector witness)

def rawBound : CoeffClass := LeftBound98640.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98640.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound98640.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound98643

namespace LeftBound98649
def owner : Owner := ⟨.program ⟨214⟩, ⟨16128⟩⟩
def transferEvent : Nat := 98649
def frameStart : Nat := 98596
def rule : BoundRule := .product (.predecessor 0 98647 .coefficient) (.predecessor 1 98648 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98647 .coefficient)
      LeftAuthority98645.bound (LeftAuthority98645.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98646RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98645.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98645.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98648 .coefficient)
      LeftBound98643.bound (LeftBound98643.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events385.exact98644RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98643.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98643.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority98645.bound LeftBound98643.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98645.bound, LeftBound98643.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority98645.actual selector witness) * (LeftBound98643.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98649

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
