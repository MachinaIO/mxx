import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard041
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard248

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound37548
def owner : Owner := ⟨.program ⟨214⟩, ⟨10043⟩⟩
def transferEvent : Nat := 37548
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨81⟩⟩]⟩ [⟨.result 8008 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8008 .coefficient)
      LeftBound8007.bound (LeftBound8007.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨81⟩⟩) (rawTerms := some (Proof.Events031.exact8008RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8007.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound8007.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound8007.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound37548

namespace LeftBound37553
def owner : Owner := ⟨.program ⟨214⟩, ⟨10044⟩⟩
def transferEvent : Nat := 37553
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 37551 .coefficient) (.predecessor 1 37552 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37551 .coefficient)
      LeftBound37547.bound (LeftBound37547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37547.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37547.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37552 .coefficient)
      LeftBound8004.bound (LeftBound8004.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8004.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8004.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound37547.bound LeftBound8004.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37547.bound, LeftBound8004.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound37547.actual selector witness) * (LeftBound8004.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37553

namespace LeftBound37554
def owner : Owner := ⟨.program ⟨214⟩, ⟨10044⟩⟩
def transferEvent : Nat := 37554
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩ [⟨.result 8001 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8001 .coefficient)
      LeftAuthority8000.bound (LeftAuthority8000.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7873⟩⟩) (rawTerms := some (Proof.Events031.exact8001RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8000.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8000.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority8000.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority8000.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound37554

namespace LeftBound37555
def owner : Owner := ⟨.program ⟨214⟩, ⟨10044⟩⟩
def transferEvent : Nat := 37555
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 37550 .summary) (.transfer 37554) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37550 .summary)
      LeftBound37548.bound (LeftBound37548.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10043⟩⟩) (rawTerms := some (Proof.Events146.exact37550RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound37548.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 37554)
      LeftBound37554.bound (LeftBound37554.actual selector witness) := by
  exact .transfer (LeftBound37554.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound37548.bound LeftBound37554.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37548.bound, LeftBound37554.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound37548.actual selector witness) * (LeftBound37554.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37555

namespace LeftBound37563
def owner : Owner := ⟨.program ⟨214⟩, ⟨12785⟩⟩
def transferEvent : Nat := 37563
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 37561 .coefficient, .predecessor 1 37562 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37561 .coefficient)
      LeftBound37553.bound (LeftBound37553.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37560RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37553.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37553.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37562 .coefficient)
      LeftBound37525.bound (LeftBound37525.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37525.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37525.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37553.bound, LeftBound37525.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37553.bound, LeftBound37525.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound37553.actual selector witness, LeftBound37525.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37563

namespace LeftBound37565
def owner : Owner := ⟨.program ⟨214⟩, ⟨12785⟩⟩
def transferEvent : Nat := 37565
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 37560 .summary, .result 37530 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37560 .summary)
      LeftBound37555.bound (LeftBound37555.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10044⟩⟩) (rawTerms := some (Proof.Events146.exact37560RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound37555.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37530 .summary)
      LeftBound37527.bound (LeftBound37527.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12784⟩⟩) (rawTerms := some (Proof.Events146.exact37530RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound37527.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37555.bound, LeftBound37527.bound]
def bound : CoeffClass := .finite ⟨95458688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37555.bound, LeftBound37527.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound37555.actual selector witness, LeftBound37527.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37565

namespace LeftBound37569
def owner : Owner := ⟨.program ⟨214⟩, ⟨25538⟩⟩
def transferEvent : Nat := 37569
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 37567 .coefficient) (.predecessor 1 37568 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37567 .coefficient)
      LeftBound37563.bound (LeftBound37563.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37566RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37563.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37563.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37568 .coefficient)
      LeftAuthority37501.bound (LeftAuthority37501.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37502RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37501.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37501.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound37563.bound LeftAuthority37501.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37563.bound, LeftAuthority37501.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound37563.actual selector witness) * (LeftAuthority37501.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37569

namespace LeftBound37570
def owner : Owner := ⟨.program ⟨214⟩, ⟨25538⟩⟩
def transferEvent : Nat := 37570
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25537⟩⟩]⟩ [⟨.result 37502 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37502 .coefficient)
      LeftAuthority37501.bound (LeftAuthority37501.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25537⟩⟩) (rawTerms := some (Proof.Events146.exact37502RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37501.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37501.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority37501.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37501.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority37501.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound37570

namespace LeftBound37571
def owner : Owner := ⟨.program ⟨214⟩, ⟨25538⟩⟩
def transferEvent : Nat := 37571
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 37566 .summary) (.transfer 37570) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37566 .summary)
      LeftBound37565.bound (LeftBound37565.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12785⟩⟩) (rawTerms := some (Proof.Events146.exact37566RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound37565.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 37570)
      LeftBound37570.bound (LeftBound37570.actual selector witness) := by
  exact .transfer (LeftBound37570.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound37565.bound LeftBound37570.bound
def bound : CoeffClass := .finite ⟨350334912299008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37565.bound, LeftBound37570.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound37565.actual selector witness) * (LeftBound37570.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37571

namespace LeftBound37582
def owner : Owner := ⟨.program ⟨214⟩, ⟨20042⟩⟩
def transferEvent : Nat := 37582
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 37580 .coefficient) (.value (.predecessor 1 37581 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37580 .coefficient)
      LeftAuthority37578.bound (LeftAuthority37578.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37579RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37578.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37578.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37581 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority37578.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37578.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority37578.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound37582

namespace LeftBound37586
def owner : Owner := ⟨.program ⟨214⟩, ⟨20043⟩⟩
def transferEvent : Nat := 37586
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 37584 .coefficient) (.predecessor 1 37585 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37584 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37585 .coefficient)
      LeftBound37582.bound (LeftBound37582.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37582.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37582.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound37582.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound37582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound37582.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37586

namespace LeftBound37587
def owner : Owner := ⟨.program ⟨214⟩, ⟨20043⟩⟩
def transferEvent : Nat := 37587
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20040⟩⟩]⟩ [⟨.result 37579 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37579 .coefficient)
      LeftAuthority37578.bound (LeftAuthority37578.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20040⟩⟩) (rawTerms := some (Proof.Events146.exact37579RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37578.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37578.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority37578.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37578.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority37578.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound37587

namespace LeftBound37588
def owner : Owner := ⟨.program ⟨214⟩, ⟨20043⟩⟩
def transferEvent : Nat := 37588
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 37587) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 37587)
      LeftBound37587.bound (LeftBound37587.actual selector witness) := by
  exact .transfer (LeftBound37587.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound37587.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound37587.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound37587.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37588

namespace LeftBound37667
def owner : Owner := ⟨.program ⟨214⟩, ⟨12779⟩⟩
def transferEvent : Nat := 37667
def frameStart : Nat := 37638
def rule : BoundRule := .product (.predecessor 0 37665 .coefficient) (.predecessor 1 37666 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37665 .coefficient)
      LeftAuthority37663.bound (LeftAuthority37663.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37664RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37663.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37663.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37666 .coefficient)
      LeftAuthority37660.bound (LeftAuthority37660.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37660.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37660.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority37663.bound LeftAuthority37660.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37663.bound, LeftAuthority37660.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority37663.actual selector witness) * (LeftAuthority37660.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37667

namespace LeftBound37671
def owner : Owner := ⟨.program ⟨214⟩, ⟨12780⟩⟩
def transferEvent : Nat := 37671
def frameStart : Nat := 37638
def rule : BoundRule := .identity (.predecessor 0 37670 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37670 .coefficient)
      LeftBound37667.bound (LeftBound37667.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events147.exact37669RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37667.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37667.derived selector witness)

def rawBound : CoeffClass := LeftBound37667.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37667.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound37667.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound37671

namespace LeftBound37688
def owner : Owner := ⟨.program ⟨214⟩, ⟨12866⟩⟩
def transferEvent : Nat := 37688
def frameStart : Nat := 37638
def rule : BoundRule := .sum [.predecessor 0 37686 .coefficient, .predecessor 1 37687 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37686 .coefficient)
      LeftBound37671.bound (LeftBound37671.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound37671.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37687 .coefficient)
      LeftAuthority37684.bound (LeftAuthority37684.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority37684.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37671.bound, LeftAuthority37684.bound]
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37671.bound, LeftAuthority37684.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound37671.actual selector witness, LeftAuthority37684.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37688

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
