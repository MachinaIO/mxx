import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard241

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound36599
def owner : Owner := ⟨.program ⟨214⟩, ⟨13177⟩⟩
def transferEvent : Nat := 36599
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 36597 .coefficient, .predecessor 1 36598 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36597 .coefficient)
      LeftBound36589.bound (LeftBound36589.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36596RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36589.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36589.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36598 .coefficient)
      LeftBound36561.bound (LeftBound36561.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36566RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36561.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36561.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36589.bound, LeftBound36561.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36589.bound, LeftBound36561.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36589.actual selector witness, LeftBound36561.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36599

namespace LeftBound36601
def owner : Owner := ⟨.program ⟨214⟩, ⟨13177⟩⟩
def transferEvent : Nat := 36601
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 36596 .summary, .result 36566 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36596 .summary)
      LeftBound36591.bound (LeftBound36591.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10254⟩⟩) (rawTerms := some (Proof.Events142.exact36596RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36591.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36566 .summary)
      LeftBound36563.bound (LeftBound36563.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13176⟩⟩) (rawTerms := some (Proof.Events142.exact36566RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36563.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36591.bound, LeftBound36563.bound]
def bound : CoeffClass := .finite ⟨95468672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36591.bound, LeftBound36563.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36591.actual selector witness, LeftBound36563.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36601

namespace LeftBound36605
def owner : Owner := ⟨.program ⟨214⟩, ⟨25692⟩⟩
def transferEvent : Nat := 36605
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 36603 .coefficient) (.predecessor 1 36604 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36603 .coefficient)
      LeftBound36599.bound (LeftBound36599.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36602RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36599.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36604 .coefficient)
      LeftAuthority36537.bound (LeftAuthority36537.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36537.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36537.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36599.bound LeftAuthority36537.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36599.bound, LeftAuthority36537.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36599.actual selector witness) * (LeftAuthority36537.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36605

namespace LeftBound36606
def owner : Owner := ⟨.program ⟨214⟩, ⟨25692⟩⟩
def transferEvent : Nat := 36606
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25691⟩⟩]⟩ [⟨.result 36538 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36538 .coefficient)
      LeftAuthority36537.bound (LeftAuthority36537.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25691⟩⟩) (rawTerms := some (Proof.Events142.exact36538RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36537.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36537.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority36537.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36537.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority36537.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound36606

namespace LeftBound36607
def owner : Owner := ⟨.program ⟨214⟩, ⟨25692⟩⟩
def transferEvent : Nat := 36607
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36602 .summary) (.transfer 36606) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36602 .summary)
      LeftBound36601.bound (LeftBound36601.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13177⟩⟩) (rawTerms := some (Proof.Events142.exact36602RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36601.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 36606)
      LeftBound36606.bound (LeftBound36606.actual selector witness) := by
  exact .transfer (LeftBound36606.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36601.bound LeftBound36606.bound
def bound : CoeffClass := .finite ⟨350371553738752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36601.bound, LeftBound36606.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36601.actual selector witness) * (LeftBound36606.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36607

namespace LeftBound36618
def owner : Owner := ⟨.program ⟨214⟩, ⟨20186⟩⟩
def transferEvent : Nat := 36618
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 36616 .coefficient) (.value (.predecessor 1 36617 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36616 .coefficient)
      LeftAuthority36614.bound (LeftAuthority36614.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36615RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36614.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36614.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36617 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority36614.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36614.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority36614.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound36618

namespace LeftBound36622
def owner : Owner := ⟨.program ⟨214⟩, ⟨20187⟩⟩
def transferEvent : Nat := 36622
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 36620 .coefficient) (.predecessor 1 36621 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36620 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36621 .coefficient)
      LeftBound36618.bound (LeftBound36618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36618.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36618.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound36618.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound36618.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound36618.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36622

namespace LeftBound36623
def owner : Owner := ⟨.program ⟨214⟩, ⟨20187⟩⟩
def transferEvent : Nat := 36623
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20184⟩⟩]⟩ [⟨.result 36615 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36615 .coefficient)
      LeftAuthority36614.bound (LeftAuthority36614.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20184⟩⟩) (rawTerms := some (Proof.Events143.exact36615RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36614.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36614.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority36614.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority36614.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound36623

namespace LeftBound36624
def owner : Owner := ⟨.program ⟨214⟩, ⟨20187⟩⟩
def transferEvent : Nat := 36624
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 36623) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 36623)
      LeftBound36623.bound (LeftBound36623.actual selector witness) := by
  exact .transfer (LeftBound36623.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound36623.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound36623.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound36623.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36624

namespace LeftBound36703
def owner : Owner := ⟨.program ⟨214⟩, ⟨13171⟩⟩
def transferEvent : Nat := 36703
def frameStart : Nat := 36674
def rule : BoundRule := .product (.predecessor 0 36701 .coefficient) (.predecessor 1 36702 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36701 .coefficient)
      LeftAuthority36699.bound (LeftAuthority36699.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36699.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36699.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36702 .coefficient)
      LeftAuthority36696.bound (LeftAuthority36696.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36697RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36696.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36696.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority36699.bound LeftAuthority36696.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36699.bound, LeftAuthority36696.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority36699.actual selector witness) * (LeftAuthority36696.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36703

namespace LeftBound36707
def owner : Owner := ⟨.program ⟨214⟩, ⟨13172⟩⟩
def transferEvent : Nat := 36707
def frameStart : Nat := 36674
def rule : BoundRule := .identity (.predecessor 0 36706 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36706 .coefficient)
      LeftBound36703.bound (LeftBound36703.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36705RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36703.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36703.derived selector witness)

def rawBound : CoeffClass := LeftBound36703.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36703.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound36703.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound36707

namespace LeftBound36724
def owner : Owner := ⟨.program ⟨214⟩, ⟨13258⟩⟩
def transferEvent : Nat := 36724
def frameStart : Nat := 36674
def rule : BoundRule := .sum [.predecessor 0 36722 .coefficient, .predecessor 1 36723 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36722 .coefficient)
      LeftBound36707.bound (LeftBound36707.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound36707.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36723 .coefficient)
      LeftAuthority36720.bound (LeftAuthority36720.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority36720.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36707.bound, LeftAuthority36720.bound]
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36707.bound, LeftAuthority36720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36707.actual selector witness, LeftAuthority36720.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36724

namespace LeftBound36727
def owner : Owner := ⟨.program ⟨214⟩, ⟨13259⟩⟩
def transferEvent : Nat := 36727
def frameStart : Nat := 36674
def rule : BoundRule := .identity (.predecessor 0 36726 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36726 .coefficient)
      LeftBound36724.bound (LeftBound36724.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound36724.derived selector witness)

def rawBound : CoeffClass := LeftBound36724.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36724.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound36724.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound36727

namespace LeftBound36733
def owner : Owner := ⟨.program ⟨214⟩, ⟨13260⟩⟩
def transferEvent : Nat := 36733
def frameStart : Nat := 36674
def rule : BoundRule := .product (.predecessor 0 36731 .coefficient) (.predecessor 1 36732 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36731 .coefficient)
      LeftAuthority36729.bound (LeftAuthority36729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36730RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36729.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36732 .coefficient)
      LeftBound36727.bound (LeftBound36727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36727.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36727.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority36729.bound LeftBound36727.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36729.bound, LeftBound36727.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority36729.actual selector witness) * (LeftBound36727.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36733

namespace LeftBound36749
def owner : Owner := ⟨.program ⟨214⟩, ⟨7880⟩⟩
def transferEvent : Nat := 36749
def frameStart : Nat := 36674
def rule : BoundRule := .scale (.predecessor 0 36747 .coefficient) (.value (.predecessor 1 36748 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36747 .coefficient)
      LeftAuthority36745.bound (LeftAuthority36745.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36746RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36745.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36745.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36748 .coefficient)
      LeftAuthority36736.bound (LeftAuthority36736.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority36736.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority36745.bound LeftAuthority36736.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36745.bound, LeftAuthority36736.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority36745.actual selector witness) * (LeftAuthority36736.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound36749

namespace LeftBound36752
def owner : Owner := ⟨.program ⟨214⟩, ⟨6769⟩⟩
def transferEvent : Nat := 36752
def frameStart : Nat := 36674
def rule : BoundRule := .identity (.predecessor 0 36751 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36751 .coefficient)
      LeftAuthority36739.bound (LeftAuthority36739.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36739.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36739.derived selector witness)

def rawBound : CoeffClass := LeftAuthority36739.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36739.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority36739.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound36752

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
