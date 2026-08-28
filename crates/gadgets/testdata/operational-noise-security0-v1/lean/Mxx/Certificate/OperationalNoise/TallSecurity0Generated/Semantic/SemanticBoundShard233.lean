import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard022
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard023
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard133
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard206
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard208
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard232

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound35649
def owner : Owner := ⟨.program ⟨214⟩, ⟨30181⟩⟩
def transferEvent : Nat := 35649
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35645 .summary, .result 31929 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35645 .summary)
      LeftBound35644.bound (LeftBound35644.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29856⟩⟩) (rawTerms := some (Proof.Events139.exact35645RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35644.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 31929 .summary)
      LeftBound31924.bound (LeftBound31924.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30180⟩⟩) (rawTerms := some (Proof.Events124.exact31929RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31924.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35644.bound, LeftBound31924.bound]
def bound : CoeffClass := .finite ⟨85361036953731453608582447156, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35644.bound, LeftBound31924.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35644.actual selector witness, LeftBound31924.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35649

namespace LeftBound35653
def owner : Owner := ⟨.program ⟨214⟩, ⟨30192⟩⟩
def transferEvent : Nat := 35653
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35651 .coefficient, .predecessor 1 35652 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35651 .coefficient)
      LeftBound35648.bound (LeftBound35648.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35650RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35648.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35648.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35652 .coefficient)
      LeftBound31710.bound (LeftBound31710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31710.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31710.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35648.bound, LeftBound31710.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35648.bound, LeftBound31710.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35648.actual selector witness, LeftBound31710.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35653

namespace LeftBound35654
def owner : Owner := ⟨.program ⟨214⟩, ⟨30192⟩⟩
def transferEvent : Nat := 35654
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35650 .summary, .result 31717 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35650 .summary)
      LeftBound35649.bound (LeftBound35649.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30181⟩⟩) (rawTerms := some (Proof.Events139.exact35650RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35649.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 31717 .summary)
      LeftBound31712.bound (LeftBound31712.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30190⟩⟩) (rawTerms := some (Proof.Events123.exact31717RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31712.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35649.bound, LeftBound31712.bound]
def bound : CoeffClass := .finite ⟨313276456757822654825721789388161076, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35649.bound, LeftBound31712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35649.actual selector witness, LeftBound31712.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35654

namespace LeftBound35660
def owner : Owner := ⟨.program ⟨214⟩, ⟨7090⟩⟩
def transferEvent : Nat := 35660
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 35658 .coefficient) (.predecessor 1 35659 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35658 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35659 .coefficient)
      LeftAuthority6003.bound (LeftAuthority6003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6003.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6003.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftAuthority6003.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftAuthority6003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftAuthority6003.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35660

namespace LeftBound35665
def owner : Owner := ⟨.program ⟨214⟩, ⟨7721⟩⟩
def transferEvent : Nat := 35665
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35663 .coefficient, .predecessor 1 35664 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35663 .coefficient)
      LeftBound35660.bound (LeftBound35660.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35662RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35660.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35660.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35664 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35660.bound, LeftBound21418.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35660.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35660.actual selector witness, LeftBound21418.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35665

namespace LeftBound35669
def owner : Owner := ⟨.program ⟨214⟩, ⟨7722⟩⟩
def transferEvent : Nat := 35669
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35667 .coefficient, .predecessor 1 35668 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35667 .coefficient)
      LeftBound35665.bound (LeftBound35665.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35666RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35665.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35665.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35668 .coefficient)
      LeftAuthority35656.bound (LeftAuthority35656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35657RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35656.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35656.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35665.bound, LeftAuthority35656.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35665.bound, LeftAuthority35656.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35665.actual selector witness, LeftAuthority35656.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35669

namespace LeftBound35670
def owner : Owner := ⟨.program ⟨214⟩, ⟨7722⟩⟩
def transferEvent : Nat := 35670
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨2⟩⟩]⟩ [⟨.result 35657 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35657 .coefficient)
      LeftAuthority35656.bound (LeftAuthority35656.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨2⟩⟩) (rawTerms := some (Proof.Events139.exact35657RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35656.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35656.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority35656.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35656.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority35656.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound35670

namespace LeftBound35675
def owner : Owner := ⟨.program ⟨214⟩, ⟨7900⟩⟩
def transferEvent : Nat := 35675
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 35673 .coefficient) (.predecessor 1 35674 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35673 .coefficient)
      LeftBound35669.bound (LeftBound35669.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35672RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35669.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35669.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35674 .coefficient)
      LeftBound5960.bound (LeftBound5960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5960.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound35669.bound LeftBound5960.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35669.bound, LeftBound5960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound35669.actual selector witness) * (LeftBound5960.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35675

namespace LeftBound35676
def owner : Owner := ⟨.program ⟨214⟩, ⟨7900⟩⟩
def transferEvent : Nat := 35676
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩ [⟨.result 5957 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5957 .coefficient)
      LeftAuthority5956.bound (LeftAuthority5956.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7885⟩⟩) (rawTerms := some (Proof.Events023.exact5957RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5956.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5956.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5956.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5956.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5956.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound35676

namespace LeftBound35677
def owner : Owner := ⟨.program ⟨214⟩, ⟨7900⟩⟩
def transferEvent : Nat := 35677
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 35672 .summary) (.transfer 35676) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35672 .summary)
      LeftBound35670.bound (LeftBound35670.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7722⟩⟩) (rawTerms := some (Proof.Events139.exact35672RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35670.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35676)
      LeftBound35676.bound (LeftBound35676.actual selector witness) := by
  exact .transfer (LeftBound35676.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound35670.bound LeftBound35676.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35670.bound, LeftBound35676.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound35670.actual selector witness) * (LeftBound35676.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35677

namespace LeftBound35703
def owner : Owner := ⟨.program ⟨214⟩, ⟨30193⟩⟩
def transferEvent : Nat := 35703
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35701 .coefficient, .predecessor 1 35702 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35701 .coefficient)
      LeftBound35675.bound (LeftBound35675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35675.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35675.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35702 .coefficient)
      LeftBound35653.bound (LeftBound35653.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35653.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35653.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35675.bound, LeftBound35653.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35675.bound, LeftBound35653.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35675.actual selector witness, LeftBound35653.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35703

namespace LeftBound35723
def owner : Owner := ⟨.program ⟨214⟩, ⟨30193⟩⟩
def transferEvent : Nat := 35723
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 35700 .summary, .result 35655 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35700 .summary)
      LeftBound35677.bound (LeftBound35677.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7900⟩⟩) (rawTerms := some (Proof.Events139.exact35700RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35677.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35655 .summary)
      LeftBound35654.bound (LeftBound35654.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30192⟩⟩) (rawTerms := some (Proof.Events139.exact35655RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35654.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35677.bound, LeftBound35654.bound]
def bound : CoeffClass := .finite ⟨313276456757822654825721789483581492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35677.bound, LeftBound35654.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35677.actual selector witness, LeftBound35654.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35723

namespace LeftBound35727
def owner : Owner := ⟨.program ⟨214⟩, ⟨30194⟩⟩
def transferEvent : Nat := 35727
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 35725 .coefficient) (.predecessor 1 35726 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35725 .coefficient)
      LeftBound35703.bound (LeftBound35703.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35724RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35703.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35703.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35726 .coefficient)
      LeftBound6000.bound (LeftBound6000.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6001RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6000.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6000.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound35703.bound LeftBound6000.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35703.bound, LeftBound6000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound35703.actual selector witness) * (LeftBound6000.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35727

namespace LeftBound35728
def owner : Owner := ⟨.program ⟨214⟩, ⟨30194⟩⟩
def transferEvent : Nat := 35728
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7821⟩⟩]⟩ [⟨.result 5997 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5997 .coefficient)
      LeftAuthority5996.bound (LeftAuthority5996.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7821⟩⟩) (rawTerms := some (Proof.Events023.exact5997RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5996.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5996.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5996.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5996.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5996.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound35728

namespace LeftBound35729
def owner : Owner := ⟨.program ⟨214⟩, ⟨30194⟩⟩
def transferEvent : Nat := 35729
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 35724 .summary) (.transfer 35728) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35724 .summary)
      LeftBound35723.bound (LeftBound35723.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30193⟩⟩) (rawTerms := some (Proof.Events139.exact35724RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35723.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35728)
      LeftBound35728.bound (LeftBound35728.actual selector witness) := by
  exact .transfer (LeftBound35728.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound35723.bound LeftBound35728.bound
def bound : CoeffClass := .finite ⟨1149729608724517268372876178953375812943872, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35723.bound, LeftBound35728.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound35723.actual selector witness) * (LeftBound35728.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35729

namespace LeftBound35791
def owner : Owner := ⟨.program ⟨214⟩, ⟨30195⟩⟩
def transferEvent : Nat := 35791
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 35789 .coefficient, .predecessor 1 35790 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35789 .coefficient)
      LeftBound35727.bound (LeftBound35727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events139.exact35788RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35727.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35727.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35790 .coefficient)
      LeftBound21308.bound (LeftBound21308.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21385RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21308.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21308.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35727.bound, LeftBound21308.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35727.bound, LeftBound21308.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35727.actual selector witness, LeftBound21308.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35791

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
