import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard361

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound53658
def owner : Owner := ⟨.program ⟨214⟩, ⟨19823⟩⟩
def transferEvent : Nat := 53658
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19820⟩⟩]⟩ [⟨.result 53650 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53650 .coefficient)
      LeftAuthority53649.bound (LeftAuthority53649.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19820⟩⟩) (rawTerms := some (Proof.Events209.exact53650RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53649.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53649.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority53649.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53649.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority53649.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound53658

namespace LeftBound53659
def owner : Owner := ⟨.program ⟨214⟩, ⟨19823⟩⟩
def transferEvent : Nat := 53659
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 53658) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 53658)
      LeftBound53658.bound (LeftBound53658.actual selector witness) := by
  exact .transfer (LeftBound53658.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound53658.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound53658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound53658.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53659

namespace LeftBound53738
def owner : Owner := ⟨.program ⟨214⟩, ⟨11966⟩⟩
def transferEvent : Nat := 53738
def frameStart : Nat := 53709
def rule : BoundRule := .product (.predecessor 0 53736 .coefficient) (.predecessor 1 53737 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53736 .coefficient)
      LeftAuthority53734.bound (LeftAuthority53734.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53735RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53734.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53734.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53737 .coefficient)
      LeftAuthority53731.bound (LeftAuthority53731.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53731.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53731.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority53734.bound LeftAuthority53731.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53734.bound, LeftAuthority53731.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority53734.actual selector witness) * (LeftAuthority53731.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53738

namespace LeftBound53742
def owner : Owner := ⟨.program ⟨214⟩, ⟨11967⟩⟩
def transferEvent : Nat := 53742
def frameStart : Nat := 53709
def rule : BoundRule := .identity (.predecessor 0 53741 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53741 .coefficient)
      LeftBound53738.bound (LeftBound53738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53738.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53738.derived selector witness)

def rawBound : CoeffClass := LeftBound53738.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53738.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound53738.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound53742

namespace LeftBound53759
def owner : Owner := ⟨.program ⟨214⟩, ⟨12057⟩⟩
def transferEvent : Nat := 53759
def frameStart : Nat := 53709
def rule : BoundRule := .sum [.predecessor 0 53757 .coefficient, .predecessor 1 53758 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53757 .coefficient)
      LeftBound53742.bound (LeftBound53742.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound53742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53758 .coefficient)
      LeftAuthority53755.bound (LeftAuthority53755.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority53755.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53742.bound, LeftAuthority53755.bound]
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53742.bound, LeftAuthority53755.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound53742.actual selector witness, LeftAuthority53755.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53759

namespace LeftBound53762
def owner : Owner := ⟨.program ⟨214⟩, ⟨12058⟩⟩
def transferEvent : Nat := 53762
def frameStart : Nat := 53709
def rule : BoundRule := .identity (.predecessor 0 53761 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53761 .coefficient)
      LeftBound53759.bound (LeftBound53759.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound53759.derived selector witness)

def rawBound : CoeffClass := LeftBound53759.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53759.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound53759.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound53762

namespace LeftBound53768
def owner : Owner := ⟨.program ⟨214⟩, ⟨12059⟩⟩
def transferEvent : Nat := 53768
def frameStart : Nat := 53709
def rule : BoundRule := .product (.predecessor 0 53766 .coefficient) (.predecessor 1 53767 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53766 .coefficient)
      LeftAuthority53764.bound (LeftAuthority53764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53765RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53764.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53764.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53767 .coefficient)
      LeftBound53762.bound (LeftBound53762.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53763RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53762.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53762.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority53764.bound LeftBound53762.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53764.bound, LeftBound53762.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority53764.actual selector witness) * (LeftBound53762.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53768

namespace LeftBound53784
def owner : Owner := ⟨.program ⟨214⟩, ⟨7865⟩⟩
def transferEvent : Nat := 53784
def frameStart : Nat := 53709
def rule : BoundRule := .scale (.predecessor 0 53782 .coefficient) (.value (.predecessor 1 53783 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53782 .coefficient)
      LeftAuthority53780.bound (LeftAuthority53780.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53781RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53780.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53780.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53783 .coefficient)
      LeftAuthority53771.bound (LeftAuthority53771.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority53771.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority53780.bound LeftAuthority53771.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53780.bound, LeftAuthority53771.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority53780.actual selector witness) * (LeftAuthority53771.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound53784

namespace LeftBound53787
def owner : Owner := ⟨.program ⟨214⟩, ⟨6764⟩⟩
def transferEvent : Nat := 53787
def frameStart : Nat := 53709
def rule : BoundRule := .identity (.predecessor 0 53786 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53786 .coefficient)
      LeftAuthority53774.bound (LeftAuthority53774.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53775RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53774.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53774.derived selector witness)

def rawBound : CoeffClass := LeftAuthority53774.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53774.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority53774.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound53787

namespace LeftBound53791
def owner : Owner := ⟨.program ⟨214⟩, ⟨7866⟩⟩
def transferEvent : Nat := 53791
def frameStart : Nat := 53709
def rule : BoundRule := .product (.predecessor 0 53789 .coefficient) (.predecessor 1 53790 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53789 .coefficient)
      LeftBound53787.bound (LeftBound53787.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53788RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53787.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53787.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53790 .coefficient)
      LeftBound53784.bound (LeftBound53784.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53785RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53784.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53784.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound53787.bound LeftBound53784.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53787.bound, LeftBound53784.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound53787.actual selector witness) * (LeftBound53784.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53791

namespace LeftBound53796
def owner : Owner := ⟨.program ⟨214⟩, ⟨12060⟩⟩
def transferEvent : Nat := 53796
def frameStart : Nat := 53709
def rule : BoundRule := .sum [.predecessor 0 53794 .coefficient, .predecessor 1 53795 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53794 .coefficient)
      LeftBound53791.bound (LeftBound53791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53793RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53791.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53791.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53795 .coefficient)
      LeftBound53768.bound (LeftBound53768.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53770RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53768.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53768.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53791.bound, LeftBound53768.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53791.bound, LeftBound53768.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound53791.actual selector witness, LeftBound53768.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53796

namespace LeftBound53800
def owner : Owner := ⟨.program ⟨214⟩, ⟨25227⟩⟩
def transferEvent : Nat := 53800
def frameStart : Nat := 53709
def rule : BoundRule := .product (.predecessor 0 53798 .coefficient) (.predecessor 1 53799 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53798 .coefficient)
      LeftBound53796.bound (LeftBound53796.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53797RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53796.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53796.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53799 .coefficient)
      LeftAuthority53753.bound (LeftAuthority53753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53754RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53753.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53753.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound53796.bound LeftAuthority53753.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53796.bound, LeftAuthority53753.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound53796.actual selector witness) * (LeftAuthority53753.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53800

namespace LeftBound53811
def owner : Owner := ⟨.program ⟨214⟩, ⟨16387⟩⟩
def transferEvent : Nat := 53811
def frameStart : Nat := 53709
def rule : BoundRule := .product (.predecessor 0 53809 .coefficient) (.predecessor 1 53810 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53809 .coefficient)
      LeftAuthority53764.bound (LeftAuthority53764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53765RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53764.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53764.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53810 .coefficient)
      LeftAuthority53807.bound (LeftAuthority53807.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53808RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53807.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53807.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority53764.bound LeftAuthority53807.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53764.bound, LeftAuthority53807.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority53764.actual selector witness) * (LeftAuthority53807.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53811

namespace LeftBound53819
def owner : Owner := ⟨.program ⟨214⟩, ⟨16388⟩⟩
def transferEvent : Nat := 53819
def frameStart : Nat := 53709
def rule : BoundRule := .sum [.predecessor 0 53817 .coefficient, .predecessor 1 53818 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53817 .coefficient)
      LeftAuthority53815.bound (LeftAuthority53815.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53816RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53815.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53815.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53818 .coefficient)
      LeftBound53811.bound (LeftBound53811.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53813RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53811.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53811.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority53815.bound, LeftBound53811.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53815.bound, LeftBound53811.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority53815.actual selector witness, LeftBound53811.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53819

namespace LeftBound53823
def owner : Owner := ⟨.program ⟨214⟩, ⟨25228⟩⟩
def transferEvent : Nat := 53823
def frameStart : Nat := 53709
def rule : BoundRule := .sum [.predecessor 0 53821 .coefficient, .predecessor 1 53822 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53821 .coefficient)
      LeftBound53819.bound (LeftBound53819.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53819.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53819.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53822 .coefficient)
      LeftBound53800.bound (LeftBound53800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53805RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53800.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53800.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53819.bound, LeftBound53800.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53819.bound, LeftBound53800.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound53819.actual selector witness, LeftBound53800.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53823

namespace LeftBound53836
def owner : Owner := ⟨.program ⟨214⟩, ⟨25226⟩⟩
def transferEvent : Nat := 53836
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 53834 .coefficient, .predecessor 1 53835 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53834 .coefficient)
      LeftBound53657.bound (LeftBound53657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53833RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53657.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53657.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53835 .coefficient)
      LeftBound53640.bound (LeftBound53640.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53647RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53640.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53640.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53657.bound, LeftBound53640.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53657.bound, LeftBound53640.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound53657.actual selector witness, LeftBound53640.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53836

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
