import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard053
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard563

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound82847
def owner : Owner := ⟨.program ⟨214⟩, ⟨7220⟩⟩
def transferEvent : Nat := 82847
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 82845 .coefficient) (.predecessor 1 82846 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82845 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82846 .coefficient)
      LeftBound9518.bound (LeftBound9518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9519RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9518.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound9518.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound9518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound9518.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82847

namespace LeftBound82852
def owner : Owner := ⟨.program ⟨214⟩, ⟨9717⟩⟩
def transferEvent : Nat := 82852
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 82850 .coefficient, .predecessor 1 82851 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82850 .coefficient)
      LeftBound82847.bound (LeftBound82847.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82849RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82847.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82847.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82851 .coefficient)
      LeftBound82842.bound (LeftBound82842.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82844RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82842.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82842.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82847.bound, LeftBound82842.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82847.bound, LeftBound82842.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82847.actual selector witness, LeftBound82842.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82852

namespace LeftBound82856
def owner : Owner := ⟨.program ⟨214⟩, ⟨9718⟩⟩
def transferEvent : Nat := 82856
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 82854 .coefficient, .predecessor 1 82855 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82854 .coefficient)
      LeftBound82852.bound (LeftBound82852.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82853RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82852.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82852.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82855 .coefficient)
      LeftBound9510.bound (LeftBound9510.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9511RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9510.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9510.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82852.bound, LeftBound9510.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82852.bound, LeftBound9510.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82852.actual selector witness, LeftBound9510.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82856

namespace LeftBound82857
def owner : Owner := ⟨.program ⟨214⟩, ⟨9718⟩⟩
def transferEvent : Nat := 82857
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨78⟩⟩]⟩ [⟨.result 9511 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9511 .coefficient)
      LeftBound9510.bound (LeftBound9510.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨78⟩⟩) (rawTerms := some (Proof.Events037.exact9511RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9510.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9510.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound9510.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9510.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound9510.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound82857

namespace LeftBound82862
def owner : Owner := ⟨.program ⟨214⟩, ⟨9719⟩⟩
def transferEvent : Nat := 82862
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 82860 .coefficient) (.predecessor 1 82861 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82860 .coefficient)
      LeftBound82856.bound (LeftBound82856.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82856.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82856.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82861 .coefficient)
      LeftBound9507.bound (LeftBound9507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9507.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9507.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound82856.bound LeftBound9507.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82856.bound, LeftBound9507.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound82856.actual selector witness) * (LeftBound9507.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82862

namespace LeftBound82863
def owner : Owner := ⟨.program ⟨214⟩, ⟨9719⟩⟩
def transferEvent : Nat := 82863
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩ [⟨.result 9504 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9504 .coefficient)
      LeftAuthority9503.bound (LeftAuthority9503.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7864⟩⟩) (rawTerms := some (Proof.Events037.exact9504RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9503.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9503.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority9503.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority9503.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound82863

namespace LeftBound82864
def owner : Owner := ⟨.program ⟨214⟩, ⟨9719⟩⟩
def transferEvent : Nat := 82864
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 82859 .summary) (.transfer 82863) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82859 .summary)
      LeftBound82857.bound (LeftBound82857.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9718⟩⟩) (rawTerms := some (Proof.Events323.exact82859RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82857.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 82863)
      LeftBound82863.bound (LeftBound82863.actual selector witness) := by
  exact .transfer (LeftBound82863.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound82857.bound LeftBound82863.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82857.bound, LeftBound82863.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound82857.actual selector witness) * (LeftBound82863.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82864

namespace LeftBound82872
def owner : Owner := ⟨.program ⟨214⟩, ⟨11964⟩⟩
def transferEvent : Nat := 82872
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 82870 .coefficient, .predecessor 1 82871 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82870 .coefficient)
      LeftBound82862.bound (LeftBound82862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82862.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82862.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82871 .coefficient)
      LeftBound82834.bound (LeftBound82834.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82834.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82834.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82862.bound, LeftBound82834.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82862.bound, LeftBound82834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82862.actual selector witness, LeftBound82834.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82872

namespace LeftBound82874
def owner : Owner := ⟨.program ⟨214⟩, ⟨11964⟩⟩
def transferEvent : Nat := 82874
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 82869 .summary, .result 82839 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82869 .summary)
      LeftBound82864.bound (LeftBound82864.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9719⟩⟩) (rawTerms := some (Proof.Events323.exact82869RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82864.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82839 .summary)
      LeftBound82836.bound (LeftBound82836.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11963⟩⟩) (rawTerms := some (Proof.Events323.exact82839RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82836.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82864.bound, LeftBound82836.bound]
def bound : CoeffClass := .finite ⟨95450368, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82864.bound, LeftBound82836.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82864.actual selector witness, LeftBound82836.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82874

namespace LeftBound82878
def owner : Owner := ⟨.program ⟨214⟩, ⟨25220⟩⟩
def transferEvent : Nat := 82878
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 82876 .coefficient) (.predecessor 1 82877 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82876 .coefficient)
      LeftBound82872.bound (LeftBound82872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82872.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82877 .coefficient)
      LeftAuthority82810.bound (LeftAuthority82810.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82811RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82810.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82810.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound82872.bound LeftAuthority82810.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82872.bound, LeftAuthority82810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound82872.actual selector witness) * (LeftAuthority82810.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82878

namespace LeftBound82879
def owner : Owner := ⟨.program ⟨214⟩, ⟨25220⟩⟩
def transferEvent : Nat := 82879
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25219⟩⟩]⟩ [⟨.result 82811 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82811 .coefficient)
      LeftAuthority82810.bound (LeftAuthority82810.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25219⟩⟩) (rawTerms := some (Proof.Events323.exact82811RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82810.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82810.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority82810.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority82810.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound82879

namespace LeftBound82880
def owner : Owner := ⟨.program ⟨214⟩, ⟨25220⟩⟩
def transferEvent : Nat := 82880
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 82875 .summary) (.transfer 82879) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82875 .summary)
      LeftBound82874.bound (LeftBound82874.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11964⟩⟩) (rawTerms := some (Proof.Events323.exact82875RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82874.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 82879)
      LeftBound82879.bound (LeftBound82879.actual selector witness) := by
  exact .transfer (LeftBound82879.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound82874.bound LeftBound82879.bound
def bound : CoeffClass := .finite ⟨350304377765888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82874.bound, LeftBound82879.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound82874.actual selector witness) * (LeftBound82879.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82880

namespace LeftBound82891
def owner : Owner := ⟨.program ⟨214⟩, ⟨19818⟩⟩
def transferEvent : Nat := 82891
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 82889 .coefficient) (.value (.predecessor 1 82890 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82889 .coefficient)
      LeftAuthority82887.bound (LeftAuthority82887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82887.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82887.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82890 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority82887.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82887.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority82887.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound82891

namespace LeftBound82895
def owner : Owner := ⟨.program ⟨214⟩, ⟨19819⟩⟩
def transferEvent : Nat := 82895
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 82893 .coefficient) (.predecessor 1 82894 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82893 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82894 .coefficient)
      LeftBound82891.bound (LeftBound82891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82891.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82891.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound82891.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound82891.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound82891.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82895

namespace LeftBound82896
def owner : Owner := ⟨.program ⟨214⟩, ⟨19819⟩⟩
def transferEvent : Nat := 82896
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19816⟩⟩]⟩ [⟨.result 82888 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82888 .coefficient)
      LeftAuthority82887.bound (LeftAuthority82887.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19816⟩⟩) (rawTerms := some (Proof.Events323.exact82888RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82887.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82887.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority82887.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82887.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority82887.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound82896

namespace LeftBound82897
def owner : Owner := ⟨.program ⟨214⟩, ⟨19819⟩⟩
def transferEvent : Nat := 82897
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 82896) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 82896)
      LeftBound82896.bound (LeftBound82896.actual selector witness) := by
  exact .transfer (LeftBound82896.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound82896.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound82896.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound82896.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82897

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
