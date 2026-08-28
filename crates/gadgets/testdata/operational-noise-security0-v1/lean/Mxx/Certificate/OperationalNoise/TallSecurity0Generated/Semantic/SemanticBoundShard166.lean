import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard165

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound25350
def owner : Owner := ⟨.program ⟨214⟩, ⟨14676⟩⟩
def transferEvent : Nat := 25350
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 25345 .summary, .result 25315 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25345 .summary)
      LeftBound25340.bound (LeftBound25340.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14675⟩⟩) (rawTerms := some (Proof.Events099.exact25345RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25340.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25315 .summary)
      LeftBound25312.bound (LeftBound25312.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14671⟩⟩) (rawTerms := some (Proof.Events098.exact25315RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25312.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25340.bound, LeftBound25312.bound]
def bound : CoeffClass := .finite ⟨95443712, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25340.bound, LeftBound25312.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25340.actual selector witness, LeftBound25312.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25350

namespace LeftBound25354
def owner : Owner := ⟨.program ⟨214⟩, ⟨26236⟩⟩
def transferEvent : Nat := 25354
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 25352 .coefficient) (.predecessor 1 25353 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25352 .coefficient)
      LeftBound25348.bound (LeftBound25348.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25351RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25348.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25348.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25353 .coefficient)
      LeftAuthority25286.bound (LeftAuthority25286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25286.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25286.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound25348.bound LeftAuthority25286.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25348.bound, LeftAuthority25286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound25348.actual selector witness) * (LeftAuthority25286.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25354

namespace LeftBound25355
def owner : Owner := ⟨.program ⟨214⟩, ⟨26236⟩⟩
def transferEvent : Nat := 25355
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26235⟩⟩]⟩ [⟨.result 25287 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25287 .coefficient)
      LeftAuthority25286.bound (LeftAuthority25286.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26235⟩⟩) (rawTerms := some (Proof.Events098.exact25287RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25286.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25286.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority25286.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority25286.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound25355

namespace LeftBound25356
def owner : Owner := ⟨.program ⟨214⟩, ⟨26236⟩⟩
def transferEvent : Nat := 25356
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 25351 .summary) (.transfer 25355) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25351 .summary)
      LeftBound25350.bound (LeftBound25350.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14676⟩⟩) (rawTerms := some (Proof.Events099.exact25351RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25350.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 25355)
      LeftBound25355.bound (LeftBound25355.actual selector witness) := by
  exact .transfer (LeftBound25355.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound25350.bound LeftBound25355.bound
def bound : CoeffClass := .finite ⟨350279950139392, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25350.bound, LeftBound25355.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound25350.actual selector witness) * (LeftBound25355.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25356

namespace LeftBound25367
def owner : Owner := ⟨.program ⟨214⟩, ⟨19686⟩⟩
def transferEvent : Nat := 25367
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 25365 .coefficient) (.value (.predecessor 1 25366 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25365 .coefficient)
      LeftAuthority25363.bound (LeftAuthority25363.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25363.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25363.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25366 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority25363.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25363.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority25363.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound25367

namespace LeftBound25371
def owner : Owner := ⟨.program ⟨214⟩, ⟨19687⟩⟩
def transferEvent : Nat := 25371
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 25369 .coefficient) (.predecessor 1 25370 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25369 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25370 .coefficient)
      LeftBound25367.bound (LeftBound25367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25368RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25367.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound25367.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound25367.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound25367.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25371

namespace LeftBound25372
def owner : Owner := ⟨.program ⟨214⟩, ⟨19687⟩⟩
def transferEvent : Nat := 25372
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19684⟩⟩]⟩ [⟨.result 25364 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25364 .coefficient)
      LeftAuthority25363.bound (LeftAuthority25363.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19684⟩⟩) (rawTerms := some (Proof.Events099.exact25364RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25363.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25363.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority25363.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25363.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority25363.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound25372

namespace LeftBound25373
def owner : Owner := ⟨.program ⟨214⟩, ⟨19687⟩⟩
def transferEvent : Nat := 25373
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 25372) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 25372)
      LeftBound25372.bound (LeftBound25372.actual selector witness) := by
  exact .transfer (LeftBound25372.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound25372.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound25372.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound25372.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25373

namespace LeftBound25452
def owner : Owner := ⟨.program ⟨214⟩, ⟨14669⟩⟩
def transferEvent : Nat := 25452
def frameStart : Nat := 25423
def rule : BoundRule := .product (.predecessor 0 25450 .coefficient) (.predecessor 1 25451 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25450 .coefficient)
      LeftAuthority25448.bound (LeftAuthority25448.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25448.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25448.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25451 .coefficient)
      LeftAuthority25445.bound (LeftAuthority25445.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25445.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25445.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority25448.bound LeftAuthority25445.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25448.bound, LeftAuthority25445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority25448.actual selector witness) * (LeftAuthority25445.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25452

namespace LeftBound25456
def owner : Owner := ⟨.program ⟨214⟩, ⟨14670⟩⟩
def transferEvent : Nat := 25456
def frameStart : Nat := 25423
def rule : BoundRule := .identity (.predecessor 0 25455 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25455 .coefficient)
      LeftBound25452.bound (LeftBound25452.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25454RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25452.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25452.derived selector witness)

def rawBound : CoeffClass := LeftBound25452.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25452.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound25452.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound25456

namespace LeftBound25473
def owner : Owner := ⟨.program ⟨214⟩, ⟨14760⟩⟩
def transferEvent : Nat := 25473
def frameStart : Nat := 25423
def rule : BoundRule := .sum [.predecessor 0 25471 .coefficient, .predecessor 1 25472 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25471 .coefficient)
      LeftBound25456.bound (LeftBound25456.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound25456.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25472 .coefficient)
      LeftAuthority25469.bound (LeftAuthority25469.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority25469.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25456.bound, LeftAuthority25469.bound]
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25456.bound, LeftAuthority25469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25456.actual selector witness, LeftAuthority25469.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25473

namespace LeftBound25476
def owner : Owner := ⟨.program ⟨214⟩, ⟨14761⟩⟩
def transferEvent : Nat := 25476
def frameStart : Nat := 25423
def rule : BoundRule := .identity (.predecessor 0 25475 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25475 .coefficient)
      LeftBound25473.bound (LeftBound25473.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound25473.derived selector witness)

def rawBound : CoeffClass := LeftBound25473.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound25473.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound25476

namespace LeftBound25482
def owner : Owner := ⟨.program ⟨214⟩, ⟨14762⟩⟩
def transferEvent : Nat := 25482
def frameStart : Nat := 25423
def rule : BoundRule := .product (.predecessor 0 25480 .coefficient) (.predecessor 1 25481 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25480 .coefficient)
      LeftAuthority25478.bound (LeftAuthority25478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25479RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25478.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25478.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25481 .coefficient)
      LeftBound25476.bound (LeftBound25476.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25477RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25476.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25476.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority25478.bound LeftBound25476.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25478.bound, LeftBound25476.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority25478.actual selector witness) * (LeftBound25476.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25482

namespace LeftBound25498
def owner : Owner := ⟨.program ⟨214⟩, ⟨7859⟩⟩
def transferEvent : Nat := 25498
def frameStart : Nat := 25423
def rule : BoundRule := .scale (.predecessor 0 25496 .coefficient) (.value (.predecessor 1 25497 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25496 .coefficient)
      LeftAuthority25494.bound (LeftAuthority25494.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25494.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25494.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25497 .coefficient)
      LeftAuthority25485.bound (LeftAuthority25485.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority25485.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority25494.bound LeftAuthority25485.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25494.bound, LeftAuthority25485.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority25494.actual selector witness) * (LeftAuthority25485.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound25498

namespace LeftBound25501
def owner : Owner := ⟨.program ⟨214⟩, ⟨6762⟩⟩
def transferEvent : Nat := 25501
def frameStart : Nat := 25423
def rule : BoundRule := .identity (.predecessor 0 25500 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25500 .coefficient)
      LeftAuthority25488.bound (LeftAuthority25488.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25489RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25488.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25488.derived selector witness)

def rawBound : CoeffClass := LeftAuthority25488.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25488.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority25488.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound25501

namespace LeftBound25505
def owner : Owner := ⟨.program ⟨214⟩, ⟨7860⟩⟩
def transferEvent : Nat := 25505
def frameStart : Nat := 25423
def rule : BoundRule := .product (.predecessor 0 25503 .coefficient) (.predecessor 1 25504 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25503 .coefficient)
      LeftBound25501.bound (LeftBound25501.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25502RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25501.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25501.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25504 .coefficient)
      LeftBound25498.bound (LeftBound25498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25499RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25498.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25498.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound25501.bound LeftBound25498.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25501.bound, LeftBound25498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound25501.actual selector witness) * (LeftBound25498.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25505

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
