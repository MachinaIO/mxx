import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard088
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard190

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound28706
def owner : Owner := ⟨.program ⟨214⟩, ⟨10860⟩⟩
def transferEvent : Nat := 28706
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 28704 .coefficient, .predecessor 1 28705 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28704 .coefficient)
      LeftBound28702.bound (LeftBound28702.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28703RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28702.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28702.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28705 .coefficient)
      LeftBound14019.bound (LeftBound14019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14019.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28702.bound, LeftBound14019.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28702.bound, LeftBound14019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28702.actual selector witness, LeftBound14019.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28706

namespace LeftBound28707
def owner : Owner := ⟨.program ⟨214⟩, ⟨10860⟩⟩
def transferEvent : Nat := 28707
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨105⟩⟩]⟩ [⟨.result 14020 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14020 .coefficient)
      LeftBound14019.bound (LeftBound14019.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨105⟩⟩) (rawTerms := some (Proof.Events054.exact14020RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14019.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound14019.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound14019.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound28707

namespace LeftBound28712
def owner : Owner := ⟨.program ⟨214⟩, ⟨10861⟩⟩
def transferEvent : Nat := 28712
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 28710 .coefficient) (.predecessor 1 28711 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28710 .coefficient)
      LeftBound28706.bound (LeftBound28706.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28709RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28706.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28706.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28711 .coefficient)
      LeftBound14016.bound (LeftBound14016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14016.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14016.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound28706.bound LeftBound14016.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28706.bound, LeftBound14016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound28706.actual selector witness) * (LeftBound14016.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28712

namespace LeftBound28713
def owner : Owner := ⟨.program ⟨214⟩, ⟨10861⟩⟩
def transferEvent : Nat := 28713
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩ [⟨.result 14013 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14013 .coefficient)
      LeftAuthority14012.bound (LeftAuthority14012.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7837⟩⟩) (rawTerms := some (Proof.Events054.exact14013RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14012.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14012.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority14012.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority14012.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound28713

namespace LeftBound28714
def owner : Owner := ⟨.program ⟨214⟩, ⟨10861⟩⟩
def transferEvent : Nat := 28714
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 28709 .summary) (.transfer 28713) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28709 .summary)
      LeftBound28707.bound (LeftBound28707.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10860⟩⟩) (rawTerms := some (Proof.Events112.exact28709RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28707.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 28713)
      LeftBound28713.bound (LeftBound28713.actual selector witness) := by
  exact .transfer (LeftBound28713.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound28707.bound LeftBound28713.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28707.bound, LeftBound28713.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound28707.actual selector witness) * (LeftBound28713.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28714

namespace LeftBound28722
def owner : Owner := ⟨.program ⟨214⟩, ⟨11008⟩⟩
def transferEvent : Nat := 28722
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 28720 .coefficient, .predecessor 1 28721 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28720 .coefficient)
      LeftBound28712.bound (LeftBound28712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28712.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28721 .coefficient)
      LeftBound28684.bound (LeftBound28684.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28684.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28684.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28712.bound, LeftBound28684.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28712.bound, LeftBound28684.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28712.actual selector witness, LeftBound28684.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28722

namespace LeftBound28724
def owner : Owner := ⟨.program ⟨214⟩, ⟨11008⟩⟩
def transferEvent : Nat := 28724
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 28719 .summary, .result 28689 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28719 .summary)
      LeftBound28714.bound (LeftBound28714.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10861⟩⟩) (rawTerms := some (Proof.Events112.exact28719RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28714.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28689 .summary)
      LeftBound28686.bound (LeftBound28686.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11007⟩⟩) (rawTerms := some (Proof.Events112.exact28689RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28686.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28714.bound, LeftBound28686.bound]
def bound : CoeffClass := .finite ⟨95423744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28714.bound, LeftBound28686.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28714.actual selector witness, LeftBound28686.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28724

namespace LeftBound28728
def owner : Owner := ⟨.program ⟨214⟩, ⟨25081⟩⟩
def transferEvent : Nat := 28728
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 28726 .coefficient) (.predecessor 1 28727 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28726 .coefficient)
      LeftBound28722.bound (LeftBound28722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28722.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28722.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28727 .coefficient)
      LeftAuthority28660.bound (LeftAuthority28660.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28660.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28660.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound28722.bound LeftAuthority28660.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28722.bound, LeftAuthority28660.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound28722.actual selector witness) * (LeftAuthority28660.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28728

namespace LeftBound28729
def owner : Owner := ⟨.program ⟨214⟩, ⟨25081⟩⟩
def transferEvent : Nat := 28729
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25080⟩⟩]⟩ [⟨.result 28661 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28661 .coefficient)
      LeftAuthority28660.bound (LeftAuthority28660.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25080⟩⟩) (rawTerms := some (Proof.Events111.exact28661RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28660.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28660.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority28660.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28660.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority28660.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound28729

namespace LeftBound28730
def owner : Owner := ⟨.program ⟨214⟩, ⟨25081⟩⟩
def transferEvent : Nat := 28730
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 28725 .summary) (.transfer 28729) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28725 .summary)
      LeftBound28724.bound (LeftBound28724.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11008⟩⟩) (rawTerms := some (Proof.Events112.exact28725RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28724.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 28729)
      LeftBound28729.bound (LeftBound28729.actual selector witness) := by
  exact .transfer (LeftBound28729.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound28724.bound LeftBound28729.bound
def bound : CoeffClass := .finite ⟨350206667259904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28724.bound, LeftBound28729.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound28724.actual selector witness) * (LeftBound28729.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28730

namespace LeftBound28741
def owner : Owner := ⟨.program ⟨214⟩, ⟨19182⟩⟩
def transferEvent : Nat := 28741
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 28739 .coefficient) (.value (.predecessor 1 28740 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28739 .coefficient)
      LeftAuthority28737.bound (LeftAuthority28737.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28738RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28737.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28737.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28740 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority28737.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28737.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority28737.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound28741

namespace LeftBound28745
def owner : Owner := ⟨.program ⟨214⟩, ⟨19183⟩⟩
def transferEvent : Nat := 28745
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 28743 .coefficient) (.predecessor 1 28744 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28743 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28744 .coefficient)
      LeftBound28741.bound (LeftBound28741.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28742RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28741.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28741.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound28741.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound28741.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound28741.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28745

namespace LeftBound28746
def owner : Owner := ⟨.program ⟨214⟩, ⟨19183⟩⟩
def transferEvent : Nat := 28746
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19180⟩⟩]⟩ [⟨.result 28738 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28738 .coefficient)
      LeftAuthority28737.bound (LeftAuthority28737.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19180⟩⟩) (rawTerms := some (Proof.Events112.exact28738RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28737.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28737.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority28737.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28737.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority28737.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound28746

namespace LeftBound28747
def owner : Owner := ⟨.program ⟨214⟩, ⟨19183⟩⟩
def transferEvent : Nat := 28747
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 28746) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 28746)
      LeftBound28746.bound (LeftBound28746.actual selector witness) := by
  exact .transfer (LeftBound28746.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound28746.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound28746.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound28746.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28747

namespace LeftBound28826
def owner : Owner := ⟨.program ⟨214⟩, ⟨11002⟩⟩
def transferEvent : Nat := 28826
def frameStart : Nat := 28797
def rule : BoundRule := .product (.predecessor 0 28824 .coefficient) (.predecessor 1 28825 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28824 .coefficient)
      LeftAuthority28822.bound (LeftAuthority28822.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28823RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28822.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28822.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28825 .coefficient)
      LeftAuthority28819.bound (LeftAuthority28819.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28819.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28819.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority28822.bound LeftAuthority28819.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28822.bound, LeftAuthority28819.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority28822.actual selector witness) * (LeftAuthority28819.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28826

namespace LeftBound28830
def owner : Owner := ⟨.program ⟨214⟩, ⟨11003⟩⟩
def transferEvent : Nat := 28830
def frameStart : Nat := 28797
def rule : BoundRule := .identity (.predecessor 0 28829 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28829 .coefficient)
      LeftBound28826.bound (LeftBound28826.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28828RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28826.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28826.derived selector witness)

def rawBound : CoeffClass := LeftBound28826.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound28826.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound28830

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
