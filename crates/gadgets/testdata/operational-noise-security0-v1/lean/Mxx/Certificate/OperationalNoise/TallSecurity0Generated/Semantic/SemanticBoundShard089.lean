import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard088

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound14046
def owner : Owner := ⟨.program ⟨214⟩, ⟨10866⟩⟩
def transferEvent : Nat := 14046
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14044 .coefficient) (.predecessor 1 14045 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14044 .coefficient)
      LeftBound14040.bound (LeftBound14040.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14043RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14040.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14040.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14045 .coefficient)
      LeftBound14016.bound (LeftBound14016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14016.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14016.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound14040.bound LeftBound14016.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14040.bound, LeftBound14016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound14040.actual selector witness) * (LeftBound14016.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14046

namespace LeftBound14047
def owner : Owner := ⟨.program ⟨214⟩, ⟨10866⟩⟩
def transferEvent : Nat := 14047
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
end LeftBound14047

namespace LeftBound14048
def owner : Owner := ⟨.program ⟨214⟩, ⟨10866⟩⟩
def transferEvent : Nat := 14048
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 14043 .summary) (.transfer 14047) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14043 .summary)
      LeftBound14041.bound (LeftBound14041.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10865⟩⟩) (rawTerms := some (Proof.Events054.exact14043RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound14041.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 14047)
      LeftBound14047.bound (LeftBound14047.actual selector witness) := by
  exact .transfer (LeftBound14047.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound14041.bound LeftBound14047.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14041.bound, LeftBound14047.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound14041.actual selector witness) * (LeftBound14047.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14048

namespace LeftBound14056
def owner : Owner := ⟨.program ⟨214⟩, ⟨11016⟩⟩
def transferEvent : Nat := 14056
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14054 .coefficient, .predecessor 1 14055 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14054 .coefficient)
      LeftBound14046.bound (LeftBound14046.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14053RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14046.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14046.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14055 .coefficient)
      LeftBound14005.bound (LeftBound14005.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14005.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14046.bound, LeftBound14005.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14046.bound, LeftBound14005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound14046.actual selector witness, LeftBound14005.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14056

namespace LeftBound14058
def owner : Owner := ⟨.program ⟨214⟩, ⟨11016⟩⟩
def transferEvent : Nat := 14058
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 14053 .summary, .result 14010 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14053 .summary)
      LeftBound14048.bound (LeftBound14048.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10866⟩⟩) (rawTerms := some (Proof.Events054.exact14053RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound14048.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14010 .summary)
      LeftBound14007.bound (LeftBound14007.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11015⟩⟩) (rawTerms := some (Proof.Events054.exact14010RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound14007.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14048.bound, LeftBound14007.bound]
def bound : CoeffClass := .finite ⟨95423744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14048.bound, LeftBound14007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound14048.actual selector witness, LeftBound14007.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14058

namespace LeftBound14062
def owner : Owner := ⟨.program ⟨214⟩, ⟨25086⟩⟩
def transferEvent : Nat := 14062
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14060 .coefficient) (.predecessor 1 14061 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14060 .coefficient)
      LeftBound14056.bound (LeftBound14056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14056.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14056.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14061 .coefficient)
      LeftAuthority13975.bound (LeftAuthority13975.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13975.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13975.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound14056.bound LeftAuthority13975.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14056.bound, LeftAuthority13975.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound14056.actual selector witness) * (LeftAuthority13975.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14062

namespace LeftBound14063
def owner : Owner := ⟨.program ⟨214⟩, ⟨25086⟩⟩
def transferEvent : Nat := 14063
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25085⟩⟩]⟩ [⟨.result 13976 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13976 .coefficient)
      LeftAuthority13975.bound (LeftAuthority13975.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25085⟩⟩) (rawTerms := some (Proof.Events054.exact13976RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13975.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13975.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority13975.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13975.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority13975.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound14063

namespace LeftBound14064
def owner : Owner := ⟨.program ⟨214⟩, ⟨25086⟩⟩
def transferEvent : Nat := 14064
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 14059 .summary) (.transfer 14063) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14059 .summary)
      LeftBound14058.bound (LeftBound14058.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11016⟩⟩) (rawTerms := some (Proof.Events054.exact14059RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound14058.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 14063)
      LeftBound14063.bound (LeftBound14063.actual selector witness) := by
  exact .transfer (LeftBound14063.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound14058.bound LeftBound14063.bound
def bound : CoeffClass := .finite ⟨350206667259904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14058.bound, LeftBound14063.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound14058.actual selector witness) * (LeftBound14063.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14064

namespace LeftBound14075
def owner : Owner := ⟨.program ⟨214⟩, ⟨19186⟩⟩
def transferEvent : Nat := 14075
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 14073 .coefficient) (.value (.predecessor 1 14074 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14073 .coefficient)
      LeftAuthority14071.bound (LeftAuthority14071.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14071.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14071.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14074 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority14071.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14071.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority14071.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound14075

namespace LeftBound14079
def owner : Owner := ⟨.program ⟨214⟩, ⟨19187⟩⟩
def transferEvent : Nat := 14079
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14077 .coefficient) (.predecessor 1 14078 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14077 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14078 .coefficient)
      LeftBound14075.bound (LeftBound14075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14075.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14075.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound14075.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound14075.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound14075.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14079

namespace LeftBound14080
def owner : Owner := ⟨.program ⟨214⟩, ⟨19187⟩⟩
def transferEvent : Nat := 14080
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19184⟩⟩]⟩ [⟨.result 14072 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14072 .coefficient)
      LeftAuthority14071.bound (LeftAuthority14071.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19184⟩⟩) (rawTerms := some (Proof.Events054.exact14072RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14071.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14071.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority14071.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14071.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority14071.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound14080

namespace LeftBound14081
def owner : Owner := ⟨.program ⟨214⟩, ⟨19187⟩⟩
def transferEvent : Nat := 14081
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 14080) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 14080)
      LeftBound14080.bound (LeftBound14080.actual selector witness) := by
  exact .transfer (LeftBound14080.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound14080.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound14080.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound14080.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14081

namespace LeftBound14160
def owner : Owner := ⟨.program ⟨214⟩, ⟨11010⟩⟩
def transferEvent : Nat := 14160
def frameStart : Nat := 14131
def rule : BoundRule := .product (.predecessor 0 14158 .coefficient) (.predecessor 1 14159 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14158 .coefficient)
      LeftAuthority14156.bound (LeftAuthority14156.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14157RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14156.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14156.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14159 .coefficient)
      LeftAuthority14153.bound (LeftAuthority14153.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14154RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14153.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14153.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority14156.bound LeftAuthority14153.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14156.bound, LeftAuthority14153.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority14156.actual selector witness) * (LeftAuthority14153.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14160

namespace LeftBound14164
def owner : Owner := ⟨.program ⟨214⟩, ⟨11011⟩⟩
def transferEvent : Nat := 14164
def frameStart : Nat := 14131
def rule : BoundRule := .identity (.predecessor 0 14163 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14163 .coefficient)
      LeftBound14160.bound (LeftBound14160.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14162RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14160.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14160.derived selector witness)

def rawBound : CoeffClass := LeftBound14160.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14160.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound14160.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound14164

namespace LeftBound14181
def owner : Owner := ⟨.program ⟨214⟩, ⟨11089⟩⟩
def transferEvent : Nat := 14181
def frameStart : Nat := 14131
def rule : BoundRule := .sum [.predecessor 0 14179 .coefficient, .predecessor 1 14180 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14179 .coefficient)
      LeftBound14164.bound (LeftBound14164.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound14164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14180 .coefficient)
      LeftAuthority14177.bound (LeftAuthority14177.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority14177.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14164.bound, LeftAuthority14177.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14164.bound, LeftAuthority14177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound14164.actual selector witness, LeftAuthority14177.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14181

namespace LeftBound14184
def owner : Owner := ⟨.program ⟨214⟩, ⟨11090⟩⟩
def transferEvent : Nat := 14184
def frameStart : Nat := 14131
def rule : BoundRule := .identity (.predecessor 0 14183 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14183 .coefficient)
      LeftBound14181.bound (LeftBound14181.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound14181.derived selector witness)

def rawBound : CoeffClass := LeftBound14181.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14181.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound14181.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound14184

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
