import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard049
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard658

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound96596
def owner : Owner := ⟨.program ⟨214⟩, ⟨9808⟩⟩
def transferEvent : Nat := 96596
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 96594 .coefficient, .predecessor 1 96595 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96594 .coefficient)
      LeftBound96592.bound (LeftBound96592.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96592.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96592.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96595 .coefficient)
      LeftBound9009.bound (LeftBound9009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9009.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96592.bound, LeftBound9009.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96592.bound, LeftBound9009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96592.actual selector witness, LeftBound9009.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96596

namespace LeftBound96597
def owner : Owner := ⟨.program ⟨214⟩, ⟨9808⟩⟩
def transferEvent : Nat := 96597
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
end LeftBound96597

namespace LeftBound96602
def owner : Owner := ⟨.program ⟨214⟩, ⟨9809⟩⟩
def transferEvent : Nat := 96602
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 96600 .coefficient) (.predecessor 1 96601 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96600 .coefficient)
      LeftBound96596.bound (LeftBound96596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96599RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96596.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96596.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96601 .coefficient)
      LeftBound9006.bound (LeftBound9006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9007RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9006.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9006.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound96596.bound LeftBound9006.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96596.bound, LeftBound9006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound96596.actual selector witness) * (LeftBound9006.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96602

namespace LeftBound96603
def owner : Owner := ⟨.program ⟨214⟩, ⟨9809⟩⟩
def transferEvent : Nat := 96603
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
end LeftBound96603

namespace LeftBound96604
def owner : Owner := ⟨.program ⟨214⟩, ⟨9809⟩⟩
def transferEvent : Nat := 96604
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 96599 .summary) (.transfer 96603) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96599 .summary)
      LeftBound96597.bound (LeftBound96597.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9808⟩⟩) (rawTerms := some (Proof.Events377.exact96599RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96597.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 96603)
      LeftBound96603.bound (LeftBound96603.actual selector witness) := by
  exact .transfer (LeftBound96603.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound96597.bound LeftBound96603.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96597.bound, LeftBound96603.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound96597.actual selector witness) * (LeftBound96603.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96604

namespace LeftBound96612
def owner : Owner := ⟨.program ⟨214⟩, ⟨12353⟩⟩
def transferEvent : Nat := 96612
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 96610 .coefficient, .predecessor 1 96611 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96610 .coefficient)
      LeftBound96602.bound (LeftBound96602.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96609RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96602.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96602.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96611 .coefficient)
      LeftBound96574.bound (LeftBound96574.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96579RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96574.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96574.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96602.bound, LeftBound96574.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96602.bound, LeftBound96574.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96602.actual selector witness, LeftBound96574.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96612

namespace LeftBound96614
def owner : Owner := ⟨.program ⟨214⟩, ⟨12353⟩⟩
def transferEvent : Nat := 96614
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 96609 .summary, .result 96579 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96609 .summary)
      LeftBound96604.bound (LeftBound96604.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9809⟩⟩) (rawTerms := some (Proof.Events377.exact96609RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96604.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96579 .summary)
      LeftBound96576.bound (LeftBound96576.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12352⟩⟩) (rawTerms := some (Proof.Events377.exact96579RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96576.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96604.bound, LeftBound96576.bound]
def bound : CoeffClass := .finite ⟨95453696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96604.bound, LeftBound96576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound96604.actual selector witness, LeftBound96576.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96614

namespace LeftBound96618
def owner : Owner := ⟨.program ⟨214⟩, ⟨25361⟩⟩
def transferEvent : Nat := 96618
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 96616 .coefficient) (.predecessor 1 96617 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96616 .coefficient)
      LeftBound96612.bound (LeftBound96612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96615RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96612.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96612.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96617 .coefficient)
      LeftAuthority96550.bound (LeftAuthority96550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96551RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96550.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96550.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound96612.bound LeftAuthority96550.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96612.bound, LeftAuthority96550.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound96612.actual selector witness) * (LeftAuthority96550.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96618

namespace LeftBound96619
def owner : Owner := ⟨.program ⟨214⟩, ⟨25361⟩⟩
def transferEvent : Nat := 96619
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25360⟩⟩]⟩ [⟨.result 96551 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96551 .coefficient)
      LeftAuthority96550.bound (LeftAuthority96550.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25360⟩⟩) (rawTerms := some (Proof.Events377.exact96551RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96550.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96550.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority96550.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96550.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority96550.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound96619

namespace LeftBound96620
def owner : Owner := ⟨.program ⟨214⟩, ⟨25361⟩⟩
def transferEvent : Nat := 96620
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 96615 .summary) (.transfer 96619) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96615 .summary)
      LeftBound96614.bound (LeftBound96614.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12353⟩⟩) (rawTerms := some (Proof.Events377.exact96615RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96614.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 96619)
      LeftBound96619.bound (LeftBound96619.actual selector witness) := by
  exact .transfer (LeftBound96619.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound96614.bound LeftBound96619.bound
def bound : CoeffClass := .finite ⟨350316591579136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96614.bound, LeftBound96619.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound96614.actual selector witness) * (LeftBound96619.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96620

namespace LeftBound96631
def owner : Owner := ⟨.program ⟨214⟩, ⟨19879⟩⟩
def transferEvent : Nat := 96631
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 96629 .coefficient) (.value (.predecessor 1 96630 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96629 .coefficient)
      LeftAuthority96627.bound (LeftAuthority96627.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96627.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96627.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96630 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority96627.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96627.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority96627.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound96631

namespace LeftBound96635
def owner : Owner := ⟨.program ⟨214⟩, ⟨19880⟩⟩
def transferEvent : Nat := 96635
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 96633 .coefficient) (.predecessor 1 96634 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96633 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96634 .coefficient)
      LeftBound96631.bound (LeftBound96631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96632RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96631.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96631.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound96631.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound96631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound96631.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96635

namespace LeftBound96636
def owner : Owner := ⟨.program ⟨214⟩, ⟨19880⟩⟩
def transferEvent : Nat := 96636
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19877⟩⟩]⟩ [⟨.result 96628 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 96628 .coefficient)
      LeftAuthority96627.bound (LeftAuthority96627.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19877⟩⟩) (rawTerms := some (Proof.Events377.exact96628RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96627.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96627.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority96627.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96627.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority96627.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound96636

namespace LeftBound96637
def owner : Owner := ⟨.program ⟨214⟩, ⟨19880⟩⟩
def transferEvent : Nat := 96637
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 96636) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 96636)
      LeftBound96636.bound (LeftBound96636.actual selector witness) := by
  exact .transfer (LeftBound96636.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound96636.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound96636.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound96636.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96637

namespace LeftBound96692
def owner : Owner := ⟨.program ⟨214⟩, ⟨12347⟩⟩
def transferEvent : Nat := 96692
def frameStart : Nat := 96675
def rule : BoundRule := .product (.predecessor 0 96690 .coefficient) (.predecessor 1 96691 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96690 .coefficient)
      LeftAuthority96688.bound (LeftAuthority96688.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96688.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96688.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 96691 .coefficient)
      LeftAuthority96685.bound (LeftAuthority96685.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96686RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96685.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96685.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority96688.bound LeftAuthority96685.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96688.bound, LeftAuthority96685.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority96688.actual selector witness) * (LeftAuthority96685.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96692

namespace LeftBound96696
def owner : Owner := ⟨.program ⟨214⟩, ⟨12348⟩⟩
def transferEvent : Nat := 96696
def frameStart : Nat := 96675
def rule : BoundRule := .identity (.predecessor 0 96695 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 96695 .coefficient)
      LeftBound96692.bound (LeftBound96692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96694RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96692.derived selector witness)

def rawBound : CoeffClass := LeftBound96692.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound96692.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound96696

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
