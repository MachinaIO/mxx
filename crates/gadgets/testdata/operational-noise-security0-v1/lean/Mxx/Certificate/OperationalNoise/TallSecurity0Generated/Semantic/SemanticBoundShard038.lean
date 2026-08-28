import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard037

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound7543
def owner : Owner := ⟨.program ⟨214⟩, ⟨12997⟩⟩
def transferEvent : Nat := 7543
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 7541 .coefficient, .predecessor 1 7542 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7541 .coefficient)
      LeftBound7533.bound (LeftBound7533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7533.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7533.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7542 .coefficient)
      LeftBound7492.bound (LeftBound7492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7497RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7492.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7492.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound7533.bound, LeftBound7492.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7533.bound, LeftBound7492.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound7533.actual selector witness, LeftBound7492.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7543

namespace LeftBound7545
def owner : Owner := ⟨.program ⟨214⟩, ⟨12997⟩⟩
def transferEvent : Nat := 7545
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 7540 .summary, .result 7497 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7540 .summary)
      LeftBound7535.bound (LeftBound7535.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10159⟩⟩) (rawTerms := some (Proof.Events029.exact7540RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound7535.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7497 .summary)
      LeftBound7494.bound (LeftBound7494.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12996⟩⟩) (rawTerms := some (Proof.Events029.exact7497RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound7494.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound7535.bound, LeftBound7494.bound]
def bound : CoeffClass := .finite ⟨95463680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7535.bound, LeftBound7494.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound7535.actual selector witness, LeftBound7494.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7545

namespace LeftBound7549
def owner : Owner := ⟨.program ⟨214⟩, ⟨25625⟩⟩
def transferEvent : Nat := 7549
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 7547 .coefficient) (.predecessor 1 7548 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7547 .coefficient)
      LeftBound7543.bound (LeftBound7543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7546RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7543.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7543.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7548 .coefficient)
      LeftAuthority7462.bound (LeftAuthority7462.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7462.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7462.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound7543.bound LeftAuthority7462.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7543.bound, LeftAuthority7462.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound7543.actual selector witness) * (LeftAuthority7462.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7549

namespace LeftBound7550
def owner : Owner := ⟨.program ⟨214⟩, ⟨25625⟩⟩
def transferEvent : Nat := 7550
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25624⟩⟩]⟩ [⟨.result 7463 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7463 .coefficient)
      LeftAuthority7462.bound (LeftAuthority7462.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25624⟩⟩) (rawTerms := some (Proof.Events029.exact7463RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7462.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7462.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority7462.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7462.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority7462.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound7550

namespace LeftBound7551
def owner : Owner := ⟨.program ⟨214⟩, ⟨25625⟩⟩
def transferEvent : Nat := 7551
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 7546 .summary) (.transfer 7550) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7546 .summary)
      LeftBound7545.bound (LeftBound7545.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12997⟩⟩) (rawTerms := some (Proof.Events029.exact7546RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound7545.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 7550)
      LeftBound7550.bound (LeftBound7550.actual selector witness) := by
  exact .transfer (LeftBound7550.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound7545.bound LeftBound7550.bound
def bound : CoeffClass := .finite ⟨350353233018880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7545.bound, LeftBound7550.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound7545.actual selector witness) * (LeftBound7550.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7551

namespace LeftBound7562
def owner : Owner := ⟨.program ⟨214⟩, ⟨20122⟩⟩
def transferEvent : Nat := 7562
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 7560 .coefficient) (.value (.predecessor 1 7561 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7560 .coefficient)
      LeftAuthority7558.bound (LeftAuthority7558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7559RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7558.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7561 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority7558.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7558.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority7558.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound7562

namespace LeftBound7566
def owner : Owner := ⟨.program ⟨214⟩, ⟨20123⟩⟩
def transferEvent : Nat := 7566
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 7564 .coefficient) (.predecessor 1 7565 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7564 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7565 .coefficient)
      LeftBound7562.bound (LeftBound7562.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7562.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7562.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound7562.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound7562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound7562.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7566

namespace LeftBound7567
def owner : Owner := ⟨.program ⟨214⟩, ⟨20123⟩⟩
def transferEvent : Nat := 7567
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20120⟩⟩]⟩ [⟨.result 7559 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7559 .coefficient)
      LeftAuthority7558.bound (LeftAuthority7558.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20120⟩⟩) (rawTerms := some (Proof.Events029.exact7559RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7558.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7558.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority7558.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority7558.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound7567

namespace LeftBound7568
def owner : Owner := ⟨.program ⟨214⟩, ⟨20123⟩⟩
def transferEvent : Nat := 7568
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 7567) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 7567)
      LeftBound7567.bound (LeftBound7567.actual selector witness) := by
  exact .transfer (LeftBound7567.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound7567.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound7567.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound7567.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7568

namespace LeftBound7647
def owner : Owner := ⟨.program ⟨214⟩, ⟨12991⟩⟩
def transferEvent : Nat := 7647
def frameStart : Nat := 7618
def rule : BoundRule := .product (.predecessor 0 7645 .coefficient) (.predecessor 1 7646 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7645 .coefficient)
      LeftAuthority7643.bound (LeftAuthority7643.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7644RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7643.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7643.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7646 .coefficient)
      LeftAuthority7640.bound (LeftAuthority7640.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7641RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7640.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7640.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority7643.bound LeftAuthority7640.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7643.bound, LeftAuthority7640.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority7643.actual selector witness) * (LeftAuthority7640.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7647

namespace LeftBound7651
def owner : Owner := ⟨.program ⟨214⟩, ⟨12992⟩⟩
def transferEvent : Nat := 7651
def frameStart : Nat := 7618
def rule : BoundRule := .identity (.predecessor 0 7650 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7650 .coefficient)
      LeftBound7647.bound (LeftBound7647.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7649RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7647.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7647.derived selector witness)

def rawBound : CoeffClass := LeftBound7647.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7647.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound7647.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound7651

namespace LeftBound7668
def owner : Owner := ⟨.program ⟨214⟩, ⟨13070⟩⟩
def transferEvent : Nat := 7668
def frameStart : Nat := 7618
def rule : BoundRule := .sum [.predecessor 0 7666 .coefficient, .predecessor 1 7667 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7666 .coefficient)
      LeftBound7651.bound (LeftBound7651.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound7651.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7667 .coefficient)
      LeftAuthority7664.bound (LeftAuthority7664.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority7664.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound7651.bound, LeftAuthority7664.bound]
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7651.bound, LeftAuthority7664.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound7651.actual selector witness, LeftAuthority7664.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound7668

namespace LeftBound7671
def owner : Owner := ⟨.program ⟨214⟩, ⟨13071⟩⟩
def transferEvent : Nat := 7671
def frameStart : Nat := 7618
def rule : BoundRule := .identity (.predecessor 0 7670 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7670 .coefficient)
      LeftBound7668.bound (LeftBound7668.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound7668.derived selector witness)

def rawBound : CoeffClass := LeftBound7668.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound7668.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound7671

namespace LeftBound7677
def owner : Owner := ⟨.program ⟨214⟩, ⟨13072⟩⟩
def transferEvent : Nat := 7677
def frameStart : Nat := 7618
def rule : BoundRule := .product (.predecessor 0 7675 .coefficient) (.predecessor 1 7676 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7675 .coefficient)
      LeftAuthority7673.bound (LeftAuthority7673.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7674RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7673.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7673.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7676 .coefficient)
      LeftBound7671.bound (LeftBound7671.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7672RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7671.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7671.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority7673.bound LeftBound7671.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7673.bound, LeftBound7671.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority7673.actual selector witness) * (LeftBound7671.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound7677

namespace LeftBound7693
def owner : Owner := ⟨.program ⟨214⟩, ⟨7877⟩⟩
def transferEvent : Nat := 7693
def frameStart : Nat := 7618
def rule : BoundRule := .scale (.predecessor 0 7691 .coefficient) (.value (.predecessor 1 7692 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7691 .coefficient)
      LeftAuthority7689.bound (LeftAuthority7689.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events030.exact7690RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7689.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7689.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 7692 .coefficient)
      LeftAuthority7680.bound (LeftAuthority7680.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority7680.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority7689.bound LeftAuthority7680.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7689.bound, LeftAuthority7680.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority7689.actual selector witness) * (LeftAuthority7680.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound7693

namespace LeftBound7696
def owner : Owner := ⟨.program ⟨214⟩, ⟨6768⟩⟩
def transferEvent : Nat := 7696
def frameStart : Nat := 7618
def rule : BoundRule := .identity (.predecessor 0 7695 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 7695 .coefficient)
      LeftAuthority7683.bound (LeftAuthority7683.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events030.exact7684RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7683.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7683.derived selector witness)

def rawBound : CoeffClass := LeftAuthority7683.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7683.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority7683.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound7696

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
