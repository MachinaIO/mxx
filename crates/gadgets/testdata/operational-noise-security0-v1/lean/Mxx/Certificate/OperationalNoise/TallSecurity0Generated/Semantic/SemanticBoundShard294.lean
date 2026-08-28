import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard292
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard293

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound43532
def owner : Owner := ⟨.program ⟨214⟩, ⟨15125⟩⟩
def transferEvent : Nat := 43532
def frameStart : Nat := 43422
def rule : BoundRule := .sum [.predecessor 0 43530 .coefficient, .predecessor 1 43531 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43530 .coefficient)
      LeftAuthority43528.bound (LeftAuthority43528.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43529RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43528.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43528.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43531 .coefficient)
      LeftBound43524.bound (LeftBound43524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43526RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43524.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43524.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority43528.bound, LeftBound43524.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43528.bound, LeftBound43524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority43528.actual selector witness, LeftBound43524.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43532

namespace LeftBound43536
def owner : Owner := ⟨.program ⟨214⟩, ⟨25079⟩⟩
def transferEvent : Nat := 43536
def frameStart : Nat := 43422
def rule : BoundRule := .sum [.predecessor 0 43534 .coefficient, .predecessor 1 43535 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43534 .coefficient)
      LeftBound43532.bound (LeftBound43532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43533RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43532.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43532.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43535 .coefficient)
      LeftBound43513.bound (LeftBound43513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43513.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43513.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43532.bound, LeftBound43513.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43532.bound, LeftBound43513.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43532.actual selector witness, LeftBound43513.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43536

namespace LeftBound43549
def owner : Owner := ⟨.program ⟨214⟩, ⟨25077⟩⟩
def transferEvent : Nat := 43549
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 43547 .coefficient, .predecessor 1 43548 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43547 .coefficient)
      LeftBound43370.bound (LeftBound43370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43546RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43370.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43370.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43548 .coefficient)
      LeftBound43353.bound (LeftBound43353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43353.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43353.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43370.bound, LeftBound43353.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43370.bound, LeftBound43353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43370.actual selector witness, LeftBound43353.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43549

namespace LeftBound43552
def owner : Owner := ⟨.program ⟨214⟩, ⟨25077⟩⟩
def transferEvent : Nat := 43552
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 43546 .summary, .result 43360 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43546 .summary)
      LeftBound43372.bound (LeftBound43372.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19179⟩⟩) (rawTerms := some (Proof.Events170.exact43546RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43372.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43360 .summary)
      LeftBound43355.bound (LeftBound43355.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25076⟩⟩) (rawTerms := some (Proof.Events169.exact43360RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43355.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43372.bound, LeftBound43355.bound]
def bound : CoeffClass := .finite ⟨352017970769920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43372.bound, LeftBound43355.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43372.actual selector witness, LeftBound43355.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43552

namespace LeftBound43556
def owner : Owner := ⟨.program ⟨214⟩, ⟨26809⟩⟩
def transferEvent : Nat := 43556
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 43554 .coefficient) (.predecessor 1 43555 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43554 .coefficient)
      LeftBound43549.bound (LeftBound43549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43549.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43549.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43555 .coefficient)
      LeftAuthority43275.bound (LeftAuthority43275.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43276RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43275.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43275.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43549.bound LeftAuthority43275.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43549.bound, LeftAuthority43275.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43549.actual selector witness) * (LeftAuthority43275.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43556

namespace LeftBound43557
def owner : Owner := ⟨.program ⟨214⟩, ⟨26809⟩⟩
def transferEvent : Nat := 43557
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26807⟩⟩]⟩ [⟨.result 43276 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43276 .coefficient)
      LeftAuthority43275.bound (LeftAuthority43275.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26807⟩⟩) (rawTerms := some (Proof.Events169.exact43276RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43275.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43275.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority43275.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43275.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority43275.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound43557

namespace LeftBound43558
def owner : Owner := ⟨.program ⟨214⟩, ⟨26809⟩⟩
def transferEvent : Nat := 43558
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 43553 .summary) (.transfer 43557) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43553 .summary)
      LeftBound43552.bound (LeftBound43552.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25077⟩⟩) (rawTerms := some (Proof.Events170.exact43553RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43552.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 43557)
      LeftBound43557.bound (LeftBound43557.actual selector witness) := by
  exact .transfer (LeftBound43557.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound43552.bound LeftBound43557.bound
def bound : CoeffClass := .finite ⟨1291911585013138718720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43552.bound, LeftBound43557.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound43552.actual selector witness) * (LeftBound43557.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43558

namespace LeftBound43569
def owner : Owner := ⟨.program ⟨214⟩, ⟨20690⟩⟩
def transferEvent : Nat := 43569
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 43567 .coefficient) (.value (.predecessor 1 43568 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43567 .coefficient)
      LeftAuthority43565.bound (LeftAuthority43565.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43566RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43565.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43565.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43568 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority43565.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43565.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority43565.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound43569

namespace LeftBound43573
def owner : Owner := ⟨.program ⟨214⟩, ⟨20691⟩⟩
def transferEvent : Nat := 43573
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 43571 .coefficient) (.predecessor 1 43572 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43571 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43572 .coefficient)
      LeftBound43569.bound (LeftBound43569.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43569.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43569.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound43569.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound43569.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound43569.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43573

namespace LeftBound43574
def owner : Owner := ⟨.program ⟨214⟩, ⟨20691⟩⟩
def transferEvent : Nat := 43574
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20688⟩⟩]⟩ [⟨.result 43566 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 43566 .coefficient)
      LeftAuthority43565.bound (LeftAuthority43565.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20688⟩⟩) (rawTerms := some (Proof.Events170.exact43566RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43565.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43565.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority43565.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43565.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority43565.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound43574

namespace LeftBound43575
def owner : Owner := ⟨.program ⟨214⟩, ⟨20691⟩⟩
def transferEvent : Nat := 43575
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 43574) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 43574)
      LeftBound43574.bound (LeftBound43574.actual selector witness) := by
  exact .transfer (LeftBound43574.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound43574.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound43574.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound43574.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43575

namespace LeftBound43670
def owner : Owner := ⟨.program ⟨214⟩, ⟨15123⟩⟩
def transferEvent : Nat := 43670
def frameStart : Nat := 43631
def rule : BoundRule := .identity (.predecessor 0 43669 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43669 .coefficient)
      LeftAuthority43667.bound (LeftAuthority43667.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43668RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43667.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43667.derived selector witness)

def rawBound : CoeffClass := LeftAuthority43667.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43667.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority43667.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound43670

namespace LeftBound43687
def owner : Owner := ⟨.program ⟨214⟩, ⟨15162⟩⟩
def transferEvent : Nat := 43687
def frameStart : Nat := 43631
def rule : BoundRule := .sum [.predecessor 0 43685 .coefficient, .predecessor 1 43686 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43685 .coefficient)
      LeftBound43670.bound (LeftBound43670.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound43670.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43686 .coefficient)
      LeftAuthority43683.bound (LeftAuthority43683.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority43683.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43670.bound, LeftAuthority43683.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43670.bound, LeftAuthority43683.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound43670.actual selector witness, LeftAuthority43683.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43687

namespace LeftBound43690
def owner : Owner := ⟨.program ⟨214⟩, ⟨15163⟩⟩
def transferEvent : Nat := 43690
def frameStart : Nat := 43631
def rule : BoundRule := .identity (.predecessor 0 43689 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43689 .coefficient)
      LeftBound43687.bound (LeftBound43687.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound43687.derived selector witness)

def rawBound : CoeffClass := LeftBound43687.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43687.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound43687.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound43690

namespace LeftBound43696
def owner : Owner := ⟨.program ⟨214⟩, ⟨15164⟩⟩
def transferEvent : Nat := 43696
def frameStart : Nat := 43631
def rule : BoundRule := .product (.predecessor 0 43694 .coefficient) (.predecessor 1 43695 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43694 .coefficient)
      LeftAuthority43692.bound (LeftAuthority43692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43692.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43692.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43695 .coefficient)
      LeftBound43690.bound (LeftBound43690.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43691RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43690.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43690.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority43692.bound LeftBound43690.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43692.bound, LeftBound43690.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority43692.actual selector witness) * (LeftBound43690.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43696

namespace LeftBound43704
def owner : Owner := ⟨.program ⟨214⟩, ⟨15165⟩⟩
def transferEvent : Nat := 43704
def frameStart : Nat := 43631
def rule : BoundRule := .sum [.predecessor 0 43702 .coefficient, .predecessor 1 43703 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 43702 .coefficient)
      LeftAuthority43700.bound (LeftAuthority43700.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43701RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43700.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43700.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 43703 .coefficient)
      LeftBound43696.bound (LeftBound43696.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43698RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43696.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43696.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority43700.bound, LeftBound43696.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43700.bound, LeftBound43696.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority43700.actual selector witness, LeftBound43696.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43704

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
