import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard049
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard458

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound67757
def owner : Owner := ⟨.program ⟨214⟩, ⟨9817⟩⟩
def transferEvent : Nat := 67757
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 67755 .coefficient, .predecessor 1 67756 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67755 .coefficient)
      LeftBound67752.bound (LeftBound67752.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67754RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67752.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67752.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67756 .coefficient)
      LeftBound67747.bound (LeftBound67747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67749RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67747.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67747.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67752.bound, LeftBound67747.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67752.bound, LeftBound67747.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67752.actual selector witness, LeftBound67747.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67757

namespace LeftBound67761
def owner : Owner := ⟨.program ⟨214⟩, ⟨9818⟩⟩
def transferEvent : Nat := 67761
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 67759 .coefficient, .predecessor 1 67760 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67759 .coefficient)
      LeftBound67757.bound (LeftBound67757.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67758RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67757.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67757.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67760 .coefficient)
      LeftBound9009.bound (LeftBound9009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9009.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67757.bound, LeftBound9009.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67757.bound, LeftBound9009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67757.actual selector witness, LeftBound9009.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67761

namespace LeftBound67762
def owner : Owner := ⟨.program ⟨214⟩, ⟨9818⟩⟩
def transferEvent : Nat := 67762
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨79⟩⟩]⟩ [⟨.result 9010 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9010 .coefficient)
      LeftBound9009.bound (LeftBound9009.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨79⟩⟩) (rawTerms := some (Proof.Events035.exact9010RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9009.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound9009.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound9009.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound67762

namespace LeftBound67767
def owner : Owner := ⟨.program ⟨214⟩, ⟨9819⟩⟩
def transferEvent : Nat := 67767
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 67765 .coefficient) (.predecessor 1 67766 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67765 .coefficient)
      LeftBound67761.bound (LeftBound67761.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67764RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67761.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67761.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67766 .coefficient)
      LeftBound9006.bound (LeftBound9006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9007RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9006.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9006.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound67761.bound LeftBound9006.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67761.bound, LeftBound9006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound67761.actual selector witness) * (LeftBound9006.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67767

namespace LeftBound67768
def owner : Owner := ⟨.program ⟨214⟩, ⟨9819⟩⟩
def transferEvent : Nat := 67768
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩ [⟨.result 9003 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9003 .coefficient)
      LeftAuthority9002.bound (LeftAuthority9002.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7867⟩⟩) (rawTerms := some (Proof.Events035.exact9003RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9002.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9002.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority9002.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9002.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority9002.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound67768

namespace LeftBound67769
def owner : Owner := ⟨.program ⟨214⟩, ⟨9819⟩⟩
def transferEvent : Nat := 67769
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 67764 .summary) (.transfer 67768) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67764 .summary)
      LeftBound67762.bound (LeftBound67762.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9818⟩⟩) (rawTerms := some (Proof.Events264.exact67764RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67762.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 67768)
      LeftBound67768.bound (LeftBound67768.actual selector witness) := by
  exact .transfer (LeftBound67768.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound67762.bound LeftBound67768.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67762.bound, LeftBound67768.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound67762.actual selector witness) * (LeftBound67768.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67769

namespace LeftBound67777
def owner : Owner := ⟨.program ⟨214⟩, ⟨12369⟩⟩
def transferEvent : Nat := 67777
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 67775 .coefficient, .predecessor 1 67776 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67775 .coefficient)
      LeftBound67767.bound (LeftBound67767.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67774RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67767.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67767.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67776 .coefficient)
      LeftBound67739.bound (LeftBound67739.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67739.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67739.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67767.bound, LeftBound67739.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67767.bound, LeftBound67739.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67767.actual selector witness, LeftBound67739.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67777

namespace LeftBound67779
def owner : Owner := ⟨.program ⟨214⟩, ⟨12369⟩⟩
def transferEvent : Nat := 67779
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 67774 .summary, .result 67744 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67774 .summary)
      LeftBound67769.bound (LeftBound67769.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9819⟩⟩) (rawTerms := some (Proof.Events264.exact67774RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67769.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67744 .summary)
      LeftBound67741.bound (LeftBound67741.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12368⟩⟩) (rawTerms := some (Proof.Events264.exact67744RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67741.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67769.bound, LeftBound67741.bound]
def bound : CoeffClass := .finite ⟨95453696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67769.bound, LeftBound67741.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67769.actual selector witness, LeftBound67741.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67779

namespace LeftBound67783
def owner : Owner := ⟨.program ⟨214⟩, ⟨25369⟩⟩
def transferEvent : Nat := 67783
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 67781 .coefficient) (.predecessor 1 67782 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67781 .coefficient)
      LeftBound67777.bound (LeftBound67777.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67777.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67777.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67782 .coefficient)
      LeftAuthority67715.bound (LeftAuthority67715.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67716RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67715.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67715.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound67777.bound LeftAuthority67715.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67777.bound, LeftAuthority67715.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound67777.actual selector witness) * (LeftAuthority67715.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67783

namespace LeftBound67784
def owner : Owner := ⟨.program ⟨214⟩, ⟨25369⟩⟩
def transferEvent : Nat := 67784
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25368⟩⟩]⟩ [⟨.result 67716 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67716 .coefficient)
      LeftAuthority67715.bound (LeftAuthority67715.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25368⟩⟩) (rawTerms := some (Proof.Events264.exact67716RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67715.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67715.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority67715.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67715.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority67715.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound67784

namespace LeftBound67785
def owner : Owner := ⟨.program ⟨214⟩, ⟨25369⟩⟩
def transferEvent : Nat := 67785
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 67780 .summary) (.transfer 67784) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67780 .summary)
      LeftBound67779.bound (LeftBound67779.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12369⟩⟩) (rawTerms := some (Proof.Events264.exact67780RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67779.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 67784)
      LeftBound67784.bound (LeftBound67784.actual selector witness) := by
  exact .transfer (LeftBound67784.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound67779.bound LeftBound67784.bound
def bound : CoeffClass := .finite ⟨350316591579136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67779.bound, LeftBound67784.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound67779.actual selector witness) * (LeftBound67784.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67785

namespace LeftBound67796
def owner : Owner := ⟨.program ⟨214⟩, ⟨19886⟩⟩
def transferEvent : Nat := 67796
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 67794 .coefficient) (.value (.predecessor 1 67795 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67794 .coefficient)
      LeftAuthority67792.bound (LeftAuthority67792.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67793RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67792.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67792.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67795 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority67792.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67792.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority67792.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound67796

namespace LeftBound67800
def owner : Owner := ⟨.program ⟨214⟩, ⟨19887⟩⟩
def transferEvent : Nat := 67800
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 67798 .coefficient) (.predecessor 1 67799 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67798 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67799 .coefficient)
      LeftBound67796.bound (LeftBound67796.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67797RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67796.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67796.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound67796.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound67796.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound67796.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67800

namespace LeftBound67801
def owner : Owner := ⟨.program ⟨214⟩, ⟨19887⟩⟩
def transferEvent : Nat := 67801
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19884⟩⟩]⟩ [⟨.result 67793 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67793 .coefficient)
      LeftAuthority67792.bound (LeftAuthority67792.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19884⟩⟩) (rawTerms := some (Proof.Events264.exact67793RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67792.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67792.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority67792.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67792.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority67792.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound67801

namespace LeftBound67802
def owner : Owner := ⟨.program ⟨214⟩, ⟨19887⟩⟩
def transferEvent : Nat := 67802
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 67801) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 67801)
      LeftBound67801.bound (LeftBound67801.actual selector witness) := by
  exact .transfer (LeftBound67801.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound67801.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound67801.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound67801.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67802

namespace LeftBound67881
def owner : Owner := ⟨.program ⟨214⟩, ⟨12363⟩⟩
def transferEvent : Nat := 67881
def frameStart : Nat := 67852
def rule : BoundRule := .product (.predecessor 0 67879 .coefficient) (.predecessor 1 67880 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67879 .coefficient)
      LeftAuthority67877.bound (LeftAuthority67877.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67877.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67877.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67880 .coefficient)
      LeftAuthority67874.bound (LeftAuthority67874.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67874.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67874.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority67877.bound LeftAuthority67874.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67877.bound, LeftAuthority67874.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority67877.actual selector witness) * (LeftAuthority67874.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67881

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
