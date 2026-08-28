import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard056
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard057
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard160

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound24756
def owner : Owner := ⟨.program ⟨214⟩, ⟨17130⟩⟩
def transferEvent : Nat := 24756
def frameStart : Nat := 24668
def rule : BoundRule := .product (.predecessor 0 24754 .coefficient) (.predecessor 1 24755 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24754 .coefficient)
      LeftAuthority24729.bound (LeftAuthority24729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24730RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24729.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24755 .coefficient)
      LeftAuthority24752.bound (LeftAuthority24752.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24753RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24752.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24752.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority24729.bound LeftAuthority24752.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24729.bound, LeftAuthority24752.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority24729.actual selector witness) * (LeftAuthority24752.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24756

namespace LeftBound24764
def owner : Owner := ⟨.program ⟨214⟩, ⟨17131⟩⟩
def transferEvent : Nat := 24764
def frameStart : Nat := 24668
def rule : BoundRule := .sum [.predecessor 0 24762 .coefficient, .predecessor 1 24763 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24762 .coefficient)
      LeftAuthority24760.bound (LeftAuthority24760.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24761RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24760.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24763 .coefficient)
      LeftBound24756.bound (LeftBound24756.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24758RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24756.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24756.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority24760.bound, LeftBound24756.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24760.bound, LeftBound24756.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority24760.actual selector witness, LeftBound24756.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24764

namespace LeftBound24768
def owner : Owner := ⟨.program ⟨214⟩, ⟨28778⟩⟩
def transferEvent : Nat := 24768
def frameStart : Nat := 24668
def rule : BoundRule := .sum [.predecessor 0 24766 .coefficient, .predecessor 1 24767 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24766 .coefficient)
      LeftBound24764.bound (LeftBound24764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24765RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24764.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24764.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24767 .coefficient)
      LeftBound24745.bound (LeftBound24745.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24745.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24745.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24764.bound, LeftBound24745.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24764.bound, LeftBound24745.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24764.actual selector witness, LeftBound24745.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24768

namespace LeftBound24781
def owner : Owner := ⟨.program ⟨214⟩, ⟨28776⟩⟩
def transferEvent : Nat := 24781
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 24779 .coefficient, .predecessor 1 24780 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24779 .coefficient)
      LeftBound24610.bound (LeftBound24610.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24610.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24610.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24780 .coefficient)
      LeftBound24593.bound (LeftBound24593.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24593.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24593.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24610.bound, LeftBound24593.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24610.bound, LeftBound24593.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24610.actual selector witness, LeftBound24593.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24781

namespace LeftBound24784
def owner : Owner := ⟨.program ⟨214⟩, ⟨28776⟩⟩
def transferEvent : Nat := 24784
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 24778 .summary, .result 24600 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24778 .summary)
      LeftBound24612.bound (LeftBound24612.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21991⟩⟩) (rawTerms := some (Proof.Events096.exact24778RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24612.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24600 .summary)
      LeftBound24595.bound (LeftBound24595.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28775⟩⟩) (rawTerms := some (Proof.Events096.exact24600RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24595.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24612.bound, LeftBound24595.bound]
def bound : CoeffClass := .finite ⟨1292270185944771604480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24612.bound, LeftBound24595.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24612.actual selector witness, LeftBound24595.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24784

namespace LeftBound24808
def owner : Owner := ⟨.program ⟨214⟩, ⟨11788⟩⟩
def transferEvent : Nat := 24808
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 24806 .coefficient) (.predecessor 1 24807 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24806 .coefficient)
      LeftAuthority1002.bound (LeftAuthority1002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact1003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1002.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1002.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24807 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1002.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1002.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1002.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound24808

namespace LeftBound24813
def owner : Owner := ⟨.program ⟨214⟩, ⟨7353⟩⟩
def transferEvent : Nat := 24813
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 24811 .coefficient) (.predecessor 1 24812 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24811 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24812 .coefficient)
      LeftBound9978.bound (LeftBound9978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9978.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound9978.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound9978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound9978.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24813

namespace LeftBound24818
def owner : Owner := ⟨.program ⟨214⟩, ⟨11789⟩⟩
def transferEvent : Nat := 24818
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 24816 .coefficient, .predecessor 1 24817 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24816 .coefficient)
      LeftBound24813.bound (LeftBound24813.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24813.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24813.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24817 .coefficient)
      LeftBound24808.bound (LeftBound24808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24810RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24808.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24808.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24813.bound, LeftBound24808.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24813.bound, LeftBound24808.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24813.actual selector witness, LeftBound24808.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24818

namespace LeftBound24822
def owner : Owner := ⟨.program ⟨214⟩, ⟨11790⟩⟩
def transferEvent : Nat := 24822
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 24820 .coefficient, .predecessor 1 24821 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24820 .coefficient)
      LeftBound24818.bound (LeftBound24818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24819RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24818.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24818.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24821 .coefficient)
      LeftBound9970.bound (LeftBound9970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9971RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9970.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24818.bound, LeftBound9970.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24818.bound, LeftBound9970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24818.actual selector witness, LeftBound9970.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24822

namespace LeftBound24823
def owner : Owner := ⟨.program ⟨214⟩, ⟨11790⟩⟩
def transferEvent : Nat := 24823
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨97⟩⟩]⟩ [⟨.result 9971 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9971 .coefficient)
      LeftBound9970.bound (LeftBound9970.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨97⟩⟩) (rawTerms := some (Proof.Events038.exact9971RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9970.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound9970.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound9970.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound24823

namespace LeftBound24828
def owner : Owner := ⟨.program ⟨214⟩, ⟨11791⟩⟩
def transferEvent : Nat := 24828
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 24826 .coefficient) (.predecessor 1 24827 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24826 .coefficient)
      LeftBound24822.bound (LeftBound24822.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events096.exact24825RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24822.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24822.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24827 .coefficient)
      LeftAuthority1005.bound (LeftAuthority1005.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact1006RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1005.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1005.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound24822.bound LeftAuthority1005.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24822.bound, LeftAuthority1005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound24822.actual selector witness) * (LeftAuthority1005.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24828

namespace LeftBound24829
def owner : Owner := ⟨.program ⟨214⟩, ⟨11791⟩⟩
def transferEvent : Nat := 24829
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9625⟩⟩], []⟩ [⟨.result 1006 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 1006 .coefficient)
      LeftAuthority1005.bound (LeftAuthority1005.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9625⟩⟩) (rawTerms := some (Proof.Events003.exact1006RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1005.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1005.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1005.bound []
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority1005.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound24829

namespace LeftBound24830
def owner : Owner := ⟨.program ⟨214⟩, ⟨11791⟩⟩
def transferEvent : Nat := 24830
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 24825 .summary) (.transfer 24829) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24825 .summary)
      LeftBound24823.bound (LeftBound24823.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11790⟩⟩) (rawTerms := some (Proof.Events096.exact24825RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24823.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 24829)
      LeftBound24829.bound (LeftBound24829.actual selector witness) := by
  exact .transfer (LeftBound24829.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound24823.bound LeftBound24829.bound
def bound : CoeffClass := .finite ⟨24960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24823.bound, LeftBound24829.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound24823.actual selector witness) * (LeftBound24829.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24830

namespace LeftBound24836
def owner : Owner := ⟨.program ⟨214⟩, ⟨9626⟩⟩
def transferEvent : Nat := 24836
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 24834 .coefficient) (.predecessor 1 24835 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24834 .coefficient)
      LeftAuthority1005.bound (LeftAuthority1005.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact1006RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1005.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1005.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24835 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1005.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1005.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1005.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound24836

namespace LeftBound24841
def owner : Owner := ⟨.program ⟨214⟩, ⟨7333⟩⟩
def transferEvent : Nat := 24841
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 24839 .coefficient) (.predecessor 1 24840 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24839 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24840 .coefficient)
      LeftBound10019.bound (LeftBound10019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10019.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound10019.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound10019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound10019.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24841

namespace LeftBound24846
def owner : Owner := ⟨.program ⟨214⟩, ⟨9627⟩⟩
def transferEvent : Nat := 24846
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 24844 .coefficient, .predecessor 1 24845 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24844 .coefficient)
      LeftBound24841.bound (LeftBound24841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact24843RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24841.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24841.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24845 .coefficient)
      LeftBound24836.bound (LeftBound24836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact24838RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24836.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24836.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24841.bound, LeftBound24836.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24841.bound, LeftBound24836.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24841.actual selector witness, LeftBound24836.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24846

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
