import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard560
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard561

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound82567
def owner : Owner := ⟨.program ⟨214⟩, ⟨16467⟩⟩
def transferEvent : Nat := 82567
def frameStart : Nat := 82467
def rule : BoundRule := .product (.predecessor 0 82565 .coefficient) (.predecessor 1 82566 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82565 .coefficient)
      LeftAuthority82522.bound (LeftAuthority82522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events322.exact82523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82522.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82522.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82566 .coefficient)
      LeftAuthority82563.bound (LeftAuthority82563.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events322.exact82564RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82563.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82563.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority82522.bound LeftAuthority82563.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82522.bound, LeftAuthority82563.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority82522.actual selector witness) * (LeftAuthority82563.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82567

namespace LeftBound82575
def owner : Owner := ⟨.program ⟨214⟩, ⟨16468⟩⟩
def transferEvent : Nat := 82575
def frameStart : Nat := 82467
def rule : BoundRule := .sum [.predecessor 0 82573 .coefficient, .predecessor 1 82574 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82573 .coefficient)
      LeftAuthority82571.bound (LeftAuthority82571.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events322.exact82572RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82571.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82571.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82574 .coefficient)
      LeftBound82567.bound (LeftBound82567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events322.exact82569RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82567.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82567.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority82571.bound, LeftBound82567.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82571.bound, LeftBound82567.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority82571.actual selector witness, LeftBound82567.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82575

namespace LeftBound82579
def owner : Owner := ⟨.program ⟨214⟩, ⟨25377⟩⟩
def transferEvent : Nat := 82579
def frameStart : Nat := 82467
def rule : BoundRule := .sum [.predecessor 0 82577 .coefficient, .predecessor 1 82578 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82577 .coefficient)
      LeftBound82575.bound (LeftBound82575.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events322.exact82576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82575.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82575.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82578 .coefficient)
      LeftBound82556.bound (LeftBound82556.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events322.exact82561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82556.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82556.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82575.bound, LeftBound82556.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82575.bound, LeftBound82556.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82575.actual selector witness, LeftBound82556.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82579

namespace LeftBound82592
def owner : Owner := ⟨.program ⟨214⟩, ⟨25375⟩⟩
def transferEvent : Nat := 82592
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 82590 .coefficient, .predecessor 1 82591 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82590 .coefficient)
      LeftBound82415.bound (LeftBound82415.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events322.exact82589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82415.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82415.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82591 .coefficient)
      LeftBound82398.bound (LeftBound82398.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82405RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82398.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82398.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82415.bound, LeftBound82398.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82415.bound, LeftBound82398.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82415.actual selector witness, LeftBound82398.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82592

namespace LeftBound82595
def owner : Owner := ⟨.program ⟨214⟩, ⟨25375⟩⟩
def transferEvent : Nat := 82595
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 82589 .summary, .result 82405 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82589 .summary)
      LeftBound82417.bound (LeftBound82417.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19891⟩⟩) (rawTerms := some (Proof.Events322.exact82589RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82417.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82405 .summary)
      LeftBound82400.bound (LeftBound82400.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25374⟩⟩) (rawTerms := some (Proof.Events321.exact82405RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82400.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82417.bound, LeftBound82400.bound]
def bound : CoeffClass := .finite ⟨352127895089152, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82417.bound, LeftBound82400.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82417.actual selector witness, LeftBound82400.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82595

namespace LeftBound82599
def owner : Owner := ⟨.program ⟨214⟩, ⟨28953⟩⟩
def transferEvent : Nat := 82599
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 82597 .coefficient) (.predecessor 1 82598 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82597 .coefficient)
      LeftBound82592.bound (LeftBound82592.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events322.exact82596RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82592.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82592.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82598 .coefficient)
      LeftAuthority82320.bound (LeftAuthority82320.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82321RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82320.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82320.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound82592.bound LeftAuthority82320.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82592.bound, LeftAuthority82320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound82592.actual selector witness) * (LeftAuthority82320.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82599

namespace LeftBound82600
def owner : Owner := ⟨.program ⟨214⟩, ⟨28953⟩⟩
def transferEvent : Nat := 82600
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28951⟩⟩]⟩ [⟨.result 82321 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82321 .coefficient)
      LeftAuthority82320.bound (LeftAuthority82320.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28951⟩⟩) (rawTerms := some (Proof.Events321.exact82321RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82320.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82320.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority82320.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority82320.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound82600

namespace LeftBound82601
def owner : Owner := ⟨.program ⟨214⟩, ⟨28953⟩⟩
def transferEvent : Nat := 82601
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 82596 .summary) (.transfer 82600) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82596 .summary)
      LeftBound82595.bound (LeftBound82595.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25375⟩⟩) (rawTerms := some (Proof.Events322.exact82596RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82595.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 82600)
      LeftBound82600.bound (LeftBound82600.actual selector witness) := by
  exact .transfer (LeftBound82600.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound82595.bound LeftBound82600.bound
def bound : CoeffClass := .finite ⟨1292315009023509266432, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82595.bound, LeftBound82600.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound82595.actual selector witness) * (LeftBound82600.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82601

namespace LeftBound82612
def owner : Owner := ⟨.program ⟨214⟩, ⟨22122⟩⟩
def transferEvent : Nat := 82612
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 82610 .coefficient) (.value (.predecessor 1 82611 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82610 .coefficient)
      LeftAuthority82608.bound (LeftAuthority82608.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events322.exact82609RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82608.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82608.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82611 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority82608.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82608.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority82608.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound82612

namespace LeftBound82616
def owner : Owner := ⟨.program ⟨214⟩, ⟨22123⟩⟩
def transferEvent : Nat := 82616
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 82614 .coefficient) (.predecessor 1 82615 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82614 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82615 .coefficient)
      LeftBound82612.bound (LeftBound82612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events322.exact82613RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82612.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82612.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound82612.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound82612.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound82612.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82616

namespace LeftBound82617
def owner : Owner := ⟨.program ⟨214⟩, ⟨22123⟩⟩
def transferEvent : Nat := 82617
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22120⟩⟩]⟩ [⟨.result 82609 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82609 .coefficient)
      LeftAuthority82608.bound (LeftAuthority82608.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22120⟩⟩) (rawTerms := some (Proof.Events322.exact82609RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82608.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82608.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority82608.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82608.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority82608.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound82617

namespace LeftBound82618
def owner : Owner := ⟨.program ⟨214⟩, ⟨22123⟩⟩
def transferEvent : Nat := 82618
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 82617) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 82617)
      LeftBound82617.bound (LeftBound82617.actual selector witness) := by
  exact .transfer (LeftBound82617.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound82617.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound82617.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound82617.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82618

namespace LeftBound82713
def owner : Owner := ⟨.program ⟨214⟩, ⟨16466⟩⟩
def transferEvent : Nat := 82713
def frameStart : Nat := 82674
def rule : BoundRule := .identity (.predecessor 0 82712 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82712 .coefficient)
      LeftAuthority82710.bound (LeftAuthority82710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82711RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82710.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82710.derived selector witness)

def rawBound : CoeffClass := LeftAuthority82710.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82710.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority82710.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound82713

namespace LeftBound82730
def owner : Owner := ⟨.program ⟨214⟩, ⟨16505⟩⟩
def transferEvent : Nat := 82730
def frameStart : Nat := 82674
def rule : BoundRule := .sum [.predecessor 0 82728 .coefficient, .predecessor 1 82729 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82728 .coefficient)
      LeftBound82713.bound (LeftBound82713.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound82713.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82729 .coefficient)
      LeftAuthority82726.bound (LeftAuthority82726.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority82726.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82713.bound, LeftAuthority82726.bound]
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82713.bound, LeftAuthority82726.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82713.actual selector witness, LeftAuthority82726.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82730

namespace LeftBound82733
def owner : Owner := ⟨.program ⟨214⟩, ⟨16506⟩⟩
def transferEvent : Nat := 82733
def frameStart : Nat := 82674
def rule : BoundRule := .identity (.predecessor 0 82732 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82732 .coefficient)
      LeftBound82730.bound (LeftBound82730.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound82730.derived selector witness)

def rawBound : CoeffClass := LeftBound82730.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82730.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound82730.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound82733

namespace LeftBound82739
def owner : Owner := ⟨.program ⟨214⟩, ⟨16507⟩⟩
def transferEvent : Nat := 82739
def frameStart : Nat := 82674
def rule : BoundRule := .product (.predecessor 0 82737 .coefficient) (.predecessor 1 82738 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82737 .coefficient)
      LeftAuthority82735.bound (LeftAuthority82735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82735.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82735.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82738 .coefficient)
      LeftBound82733.bound (LeftBound82733.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events323.exact82734RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82733.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82733.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority82735.bound LeftBound82733.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82735.bound, LeftBound82733.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority82735.actual selector witness) * (LeftBound82733.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82739

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
