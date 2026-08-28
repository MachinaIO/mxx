import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard098
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard126

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound20643
def owner : Owner := ⟨.program ⟨214⟩, ⟨26610⟩⟩
def transferEvent : Nat := 20643
def frameStart : Nat := 20566
def rule : BoundRule := .product (.predecessor 0 20641 .coefficient) (.predecessor 1 20642 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20641 .coefficient)
      LeftBound20639.bound (LeftBound20639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20640RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20639.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20639.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20642 .coefficient)
      LeftAuthority20616.bound (LeftAuthority20616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20617RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20616.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20616.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound20639.bound LeftAuthority20616.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20639.bound, LeftAuthority20616.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound20639.actual selector witness) * (LeftAuthority20616.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20643

namespace LeftBound20654
def owner : Owner := ⟨.program ⟨214⟩, ⟨15070⟩⟩
def transferEvent : Nat := 20654
def frameStart : Nat := 20566
def rule : BoundRule := .product (.predecessor 0 20652 .coefficient) (.predecessor 1 20653 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20652 .coefficient)
      LeftAuthority20627.bound (LeftAuthority20627.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20627.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20627.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20653 .coefficient)
      LeftAuthority20650.bound (LeftAuthority20650.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20651RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20650.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20650.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority20627.bound LeftAuthority20650.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20627.bound, LeftAuthority20650.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority20627.actual selector witness) * (LeftAuthority20650.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20654

namespace LeftBound20662
def owner : Owner := ⟨.program ⟨214⟩, ⟨15071⟩⟩
def transferEvent : Nat := 20662
def frameStart : Nat := 20566
def rule : BoundRule := .sum [.predecessor 0 20660 .coefficient, .predecessor 1 20661 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20660 .coefficient)
      LeftAuthority20658.bound (LeftAuthority20658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20658.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20658.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20661 .coefficient)
      LeftBound20654.bound (LeftBound20654.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20656RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20654.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20654.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority20658.bound, LeftBound20654.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20658.bound, LeftBound20654.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority20658.actual selector witness, LeftBound20654.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20662

namespace LeftBound20666
def owner : Owner := ⟨.program ⟨214⟩, ⟨26615⟩⟩
def transferEvent : Nat := 20666
def frameStart : Nat := 20566
def rule : BoundRule := .sum [.predecessor 0 20664 .coefficient, .predecessor 1 20665 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20664 .coefficient)
      LeftBound20662.bound (LeftBound20662.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20662.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20662.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20665 .coefficient)
      LeftBound20643.bound (LeftBound20643.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20648RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20643.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20643.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20662.bound, LeftBound20643.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20662.bound, LeftBound20643.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20662.actual selector witness, LeftBound20643.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20666

namespace LeftBound20679
def owner : Owner := ⟨.program ⟨214⟩, ⟨26612⟩⟩
def transferEvent : Nat := 20679
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 20677 .coefficient, .predecessor 1 20678 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20677 .coefficient)
      LeftBound20508.bound (LeftBound20508.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20676RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20508.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20508.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20678 .coefficient)
      LeftBound20491.bound (LeftBound20491.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20491.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20491.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20508.bound, LeftBound20491.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20508.bound, LeftBound20491.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20508.actual selector witness, LeftBound20491.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20679

namespace LeftBound20682
def owner : Owner := ⟨.program ⟨214⟩, ⟨26612⟩⟩
def transferEvent : Nat := 20682
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 20676 .summary, .result 20498 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20676 .summary)
      LeftBound20510.bound (LeftBound20510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20483⟩⟩) (rawTerms := some (Proof.Events080.exact20676RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20498 .summary)
      LeftBound20493.bound (LeftBound20493.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26611⟩⟩) (rawTerms := some (Proof.Events080.exact20498RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20493.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20510.bound, LeftBound20493.bound]
def bound : CoeffClass := .finite ⟨1291900380601931935744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20510.bound, LeftBound20493.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20510.actual selector witness, LeftBound20493.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20682

namespace LeftBound20686
def owner : Owner := ⟨.program ⟨214⟩, ⟨26613⟩⟩
def transferEvent : Nat := 20686
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 20684 .coefficient) (.predecessor 1 20685 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20684 .coefficient)
      LeftBound20679.bound (LeftBound20679.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20679.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20679.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20685 .coefficient)
      LeftBound5838.bound (LeftBound5838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5838.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5838.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound20679.bound LeftBound5838.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20679.bound, LeftBound5838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound20679.actual selector witness) * (LeftBound5838.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20686

namespace LeftBound20687
def owner : Owner := ⟨.program ⟨214⟩, ⟨26613⟩⟩
def transferEvent : Nat := 20687
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩ [⟨.result 5835 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5835 .coefficient)
      LeftAuthority5834.bound (LeftAuthority5834.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6671⟩⟩) (rawTerms := some (Proof.Events022.exact5835RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5834.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5834.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5834.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5834.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound20687

namespace LeftBound20688
def owner : Owner := ⟨.program ⟨214⟩, ⟨26613⟩⟩
def transferEvent : Nat := 20688
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 20683 .summary) (.transfer 20687) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20683 .summary)
      LeftBound20682.bound (LeftBound20682.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26612⟩⟩) (rawTerms := some (Proof.Events080.exact20683RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20682.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 20687)
      LeftBound20687.bound (LeftBound20687.actual selector witness) := by
  exact .transfer (LeftBound20687.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound20682.bound LeftBound20687.bound
def bound : CoeffClass := .finite ⟨4741295067215179835091451904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20682.bound, LeftBound20687.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound20682.actual selector witness) * (LeftBound20687.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20688

namespace LeftBound20703
def owner : Owner := ⟨.program ⟨214⟩, ⟨26401⟩⟩
def transferEvent : Nat := 20703
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 20701 .coefficient) (.predecessor 1 20702 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20701 .coefficient)
      LeftBound15260.bound (LeftBound15260.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events059.exact15264RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15260.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15260.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20702 .coefficient)
      LeftAuthority20699.bound (LeftAuthority20699.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20699.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20699.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound15260.bound LeftAuthority20699.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15260.bound, LeftAuthority20699.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound15260.actual selector witness) * (LeftAuthority20699.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20703

namespace LeftBound20704
def owner : Owner := ⟨.program ⟨214⟩, ⟨26401⟩⟩
def transferEvent : Nat := 20704
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26399⟩⟩]⟩ [⟨.result 20700 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20700 .coefficient)
      LeftAuthority20699.bound (LeftAuthority20699.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26399⟩⟩) (rawTerms := some (Proof.Events080.exact20700RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20699.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20699.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority20699.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20699.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority20699.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound20704

namespace LeftBound20705
def owner : Owner := ⟨.program ⟨214⟩, ⟨26401⟩⟩
def transferEvent : Nat := 20705
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 15264 .summary) (.transfer 20704) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15264 .summary)
      LeftBound15263.bound (LeftBound15263.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24933⟩⟩) (rawTerms := some (Proof.Events059.exact15264RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound15263.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 20704)
      LeftBound20704.bound (LeftBound20704.actual selector witness) := by
  exact .transfer (LeftBound20704.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound15263.bound LeftBound20704.bound
def bound : CoeffClass := .finite ⟨1291889172568118132736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15263.bound, LeftBound20704.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound15263.actual selector witness) * (LeftBound20704.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20705

namespace LeftBound20716
def owner : Owner := ⟨.program ⟨214⟩, ⟨20338⟩⟩
def transferEvent : Nat := 20716
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 20714 .coefficient) (.value (.predecessor 1 20715 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20714 .coefficient)
      LeftAuthority20712.bound (LeftAuthority20712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20712.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20715 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority20712.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20712.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority20712.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound20716

namespace LeftBound20720
def owner : Owner := ⟨.program ⟨214⟩, ⟨20339⟩⟩
def transferEvent : Nat := 20720
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 20718 .coefficient) (.predecessor 1 20719 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20718 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20719 .coefficient)
      LeftBound20716.bound (LeftBound20716.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20716.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20716.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound20716.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound20716.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound20716.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20720

namespace LeftBound20721
def owner : Owner := ⟨.program ⟨214⟩, ⟨20339⟩⟩
def transferEvent : Nat := 20721
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20336⟩⟩]⟩ [⟨.result 20713 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20713 .coefficient)
      LeftAuthority20712.bound (LeftAuthority20712.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20336⟩⟩) (rawTerms := some (Proof.Events080.exact20713RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20712.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20712.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority20712.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority20712.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound20721

namespace LeftBound20722
def owner : Owner := ⟨.program ⟨214⟩, ⟨20339⟩⟩
def transferEvent : Nat := 20722
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 20721) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 20721)
      LeftBound20721.bound (LeftBound20721.actual selector witness) := by
  exact .transfer (LeftBound20721.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound20721.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound20721.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound20721.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20722

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
