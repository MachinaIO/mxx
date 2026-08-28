import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard076
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard382

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound56510
def owner : Owner := ⟨.program ⟨214⟩, ⟨13788⟩⟩
def transferEvent : Nat := 56510
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 56508 .coefficient, .predecessor 1 56509 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56508 .coefficient)
      LeftBound56506.bound (LeftBound56506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56507RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56506.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56509 .coefficient)
      LeftBound12516.bound (LeftBound12516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12516.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56506.bound, LeftBound12516.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56506.bound, LeftBound12516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56506.actual selector witness, LeftBound12516.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56510

namespace LeftBound56511
def owner : Owner := ⟨.program ⟨214⟩, ⟨13788⟩⟩
def transferEvent : Nat := 56511
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨108⟩⟩]⟩ [⟨.result 12517 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12517 .coefficient)
      LeftBound12516.bound (LeftBound12516.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨108⟩⟩) (rawTerms := some (Proof.Events048.exact12517RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12516.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound12516.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound12516.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound56511

namespace LeftBound56516
def owner : Owner := ⟨.program ⟨214⟩, ⟨13789⟩⟩
def transferEvent : Nat := 56516
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 56514 .coefficient) (.predecessor 1 56515 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56514 .coefficient)
      LeftBound56510.bound (LeftBound56510.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56510.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56515 .coefficient)
      LeftBound12513.bound (LeftBound12513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12514RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12513.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12513.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56510.bound LeftBound12513.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56510.bound, LeftBound12513.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56510.actual selector witness) * (LeftBound12513.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56516

namespace LeftBound56517
def owner : Owner := ⟨.program ⟨214⟩, ⟨13789⟩⟩
def transferEvent : Nat := 56517
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩ [⟨.result 12510 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12510 .coefficient)
      LeftAuthority12509.bound (LeftAuthority12509.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7846⟩⟩) (rawTerms := some (Proof.Events048.exact12510RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12509.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12509.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority12509.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12509.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority12509.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound56517

namespace LeftBound56518
def owner : Owner := ⟨.program ⟨214⟩, ⟨13789⟩⟩
def transferEvent : Nat := 56518
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 56513 .summary) (.transfer 56517) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56513 .summary)
      LeftBound56511.bound (LeftBound56511.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13788⟩⟩) (rawTerms := some (Proof.Events220.exact56513RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56511.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 56517)
      LeftBound56517.bound (LeftBound56517.actual selector witness) := by
  exact .transfer (LeftBound56517.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56511.bound LeftBound56517.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56511.bound, LeftBound56517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56511.actual selector witness) * (LeftBound56517.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56518

namespace LeftBound56526
def owner : Owner := ⟨.program ⟨214⟩, ⟨13790⟩⟩
def transferEvent : Nat := 56526
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 56524 .coefficient, .predecessor 1 56525 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56524 .coefficient)
      LeftBound56516.bound (LeftBound56516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56516.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56525 .coefficient)
      LeftBound56488.bound (LeftBound56488.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56493RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56488.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56488.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56516.bound, LeftBound56488.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56516.bound, LeftBound56488.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56516.actual selector witness, LeftBound56488.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56526

namespace LeftBound56528
def owner : Owner := ⟨.program ⟨214⟩, ⟨13790⟩⟩
def transferEvent : Nat := 56528
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 56523 .summary, .result 56493 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56523 .summary)
      LeftBound56518.bound (LeftBound56518.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13789⟩⟩) (rawTerms := some (Proof.Events220.exact56523RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56518.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56493 .summary)
      LeftBound56490.bound (LeftBound56490.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13785⟩⟩) (rawTerms := some (Proof.Events220.exact56493RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56490.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56518.bound, LeftBound56490.bound]
def bound : CoeffClass := .finite ⟨95430400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56518.bound, LeftBound56490.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56518.actual selector witness, LeftBound56490.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56528

namespace LeftBound56532
def owner : Owner := ⟨.program ⟨214⟩, ⟨25918⟩⟩
def transferEvent : Nat := 56532
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 56530 .coefficient) (.predecessor 1 56531 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56530 .coefficient)
      LeftBound56526.bound (LeftBound56526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56529RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56526.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56531 .coefficient)
      LeftAuthority56464.bound (LeftAuthority56464.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56465RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56464.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56464.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56526.bound LeftAuthority56464.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56526.bound, LeftAuthority56464.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56526.actual selector witness) * (LeftAuthority56464.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56532

namespace LeftBound56533
def owner : Owner := ⟨.program ⟨214⟩, ⟨25918⟩⟩
def transferEvent : Nat := 56533
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25917⟩⟩]⟩ [⟨.result 56465 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56465 .coefficient)
      LeftAuthority56464.bound (LeftAuthority56464.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25917⟩⟩) (rawTerms := some (Proof.Events220.exact56465RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56464.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56464.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority56464.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56464.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority56464.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound56533

namespace LeftBound56534
def owner : Owner := ⟨.program ⟨214⟩, ⟨25918⟩⟩
def transferEvent : Nat := 56534
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 56529 .summary) (.transfer 56533) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56529 .summary)
      LeftBound56528.bound (LeftBound56528.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13790⟩⟩) (rawTerms := some (Proof.Events220.exact56529RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56528.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 56533)
      LeftBound56533.bound (LeftBound56533.actual selector witness) := by
  exact .transfer (LeftBound56533.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56528.bound LeftBound56533.bound
def bound : CoeffClass := .finite ⟨350231094886400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56528.bound, LeftBound56533.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56528.actual selector witness) * (LeftBound56533.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56534

namespace LeftBound56545
def owner : Owner := ⟨.program ⟨214⟩, ⟨19390⟩⟩
def transferEvent : Nat := 56545
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 56543 .coefficient) (.value (.predecessor 1 56544 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56543 .coefficient)
      LeftAuthority56541.bound (LeftAuthority56541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56542RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56541.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56541.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56544 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority56541.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56541.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority56541.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound56545

namespace LeftBound56549
def owner : Owner := ⟨.program ⟨214⟩, ⟨19391⟩⟩
def transferEvent : Nat := 56549
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 56547 .coefficient) (.predecessor 1 56548 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56547 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56548 .coefficient)
      LeftBound56545.bound (LeftBound56545.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56546RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56545.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56545.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound56545.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound56545.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound56545.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56549

namespace LeftBound56550
def owner : Owner := ⟨.program ⟨214⟩, ⟨19391⟩⟩
def transferEvent : Nat := 56550
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19388⟩⟩]⟩ [⟨.result 56542 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56542 .coefficient)
      LeftAuthority56541.bound (LeftAuthority56541.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19388⟩⟩) (rawTerms := some (Proof.Events220.exact56542RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56541.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56541.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority56541.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority56541.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound56550

namespace LeftBound56551
def owner : Owner := ⟨.program ⟨214⟩, ⟨19391⟩⟩
def transferEvent : Nat := 56551
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 56550) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 56550)
      LeftBound56550.bound (LeftBound56550.actual selector witness) := by
  exact .transfer (LeftBound56550.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound56550.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound56550.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound56550.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56551

namespace LeftBound56630
def owner : Owner := ⟨.program ⟨214⟩, ⟨13783⟩⟩
def transferEvent : Nat := 56630
def frameStart : Nat := 56601
def rule : BoundRule := .product (.predecessor 0 56628 .coefficient) (.predecessor 1 56629 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56628 .coefficient)
      LeftAuthority56626.bound (LeftAuthority56626.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56627RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56626.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56626.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56629 .coefficient)
      LeftAuthority56623.bound (LeftAuthority56623.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56624RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56623.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56623.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority56626.bound LeftAuthority56623.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56626.bound, LeftAuthority56623.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority56626.actual selector witness) * (LeftAuthority56623.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56630

namespace LeftBound56634
def owner : Owner := ⟨.program ⟨214⟩, ⟨13784⟩⟩
def transferEvent : Nat := 56634
def frameStart : Nat := 56601
def rule : BoundRule := .identity (.predecessor 0 56633 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56633 .coefficient)
      LeftBound56630.bound (LeftBound56630.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56632RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56630.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56630.derived selector witness)

def rawBound : CoeffClass := LeftBound56630.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56630.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound56630.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound56634

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
