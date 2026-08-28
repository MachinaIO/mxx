import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard502

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound73567
def owner : Owner := ⟨.program ⟨214⟩, ⟨24907⟩⟩
def transferEvent : Nat := 73567
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 73565 .coefficient) (.predecessor 1 73566 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73565 .coefficient)
      LeftBound73561.bound (LeftBound73561.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73564RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73561.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73561.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73566 .coefficient)
      LeftAuthority73499.bound (LeftAuthority73499.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73499.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73499.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound73561.bound LeftAuthority73499.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73561.bound, LeftAuthority73499.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound73561.actual selector witness) * (LeftAuthority73499.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73567

namespace LeftBound73568
def owner : Owner := ⟨.program ⟨214⟩, ⟨24907⟩⟩
def transferEvent : Nat := 73568
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨24906⟩⟩]⟩ [⟨.result 73500 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73500 .coefficient)
      LeftAuthority73499.bound (LeftAuthority73499.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨24906⟩⟩) (rawTerms := some (Proof.Events287.exact73500RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73499.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73499.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority73499.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73499.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority73499.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound73568

namespace LeftBound73569
def owner : Owner := ⟨.program ⟨214⟩, ⟨24907⟩⟩
def transferEvent : Nat := 73569
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 73564 .summary) (.transfer 73568) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73564 .summary)
      LeftBound73563.bound (LeftBound73563.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10479⟩⟩) (rawTerms := some (Proof.Events287.exact73564RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73563.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 73568)
      LeftBound73568.bound (LeftBound73568.actual selector witness) := by
  exact .transfer (LeftBound73568.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound73563.bound LeftBound73568.bound
def bound : CoeffClass := .finite ⟨350200560353280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73563.bound, LeftBound73568.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound73563.actual selector witness) * (LeftBound73568.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73569

namespace LeftBound73580
def owner : Owner := ⟨.program ⟨214⟩, ⟨19022⟩⟩
def transferEvent : Nat := 73580
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 73578 .coefficient) (.value (.predecessor 1 73579 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73578 .coefficient)
      LeftAuthority73576.bound (LeftAuthority73576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73577RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73576.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73576.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73579 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority73576.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73576.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority73576.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound73580

namespace LeftBound73584
def owner : Owner := ⟨.program ⟨214⟩, ⟨19023⟩⟩
def transferEvent : Nat := 73584
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 73582 .coefficient) (.predecessor 1 73583 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73582 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73583 .coefficient)
      LeftBound73580.bound (LeftBound73580.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73580.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73580.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound73580.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound73580.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound73580.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73584

namespace LeftBound73585
def owner : Owner := ⟨.program ⟨214⟩, ⟨19023⟩⟩
def transferEvent : Nat := 73585
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19020⟩⟩]⟩ [⟨.result 73577 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73577 .coefficient)
      LeftAuthority73576.bound (LeftAuthority73576.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19020⟩⟩) (rawTerms := some (Proof.Events287.exact73577RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73576.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73576.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority73576.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority73576.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound73585

namespace LeftBound73586
def owner : Owner := ⟨.program ⟨214⟩, ⟨19023⟩⟩
def transferEvent : Nat := 73586
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 73585) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 73585)
      LeftBound73585.bound (LeftBound73585.actual selector witness) := by
  exact .transfer (LeftBound73585.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound73585.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound73585.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound73585.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73586

namespace LeftBound73665
def owner : Owner := ⟨.program ⟨214⟩, ⟨10473⟩⟩
def transferEvent : Nat := 73665
def frameStart : Nat := 73636
def rule : BoundRule := .product (.predecessor 0 73663 .coefficient) (.predecessor 1 73664 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73663 .coefficient)
      LeftAuthority73661.bound (LeftAuthority73661.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73662RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73661.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73661.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73664 .coefficient)
      LeftAuthority73658.bound (LeftAuthority73658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73658.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73658.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority73661.bound LeftAuthority73658.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73661.bound, LeftAuthority73658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority73661.actual selector witness) * (LeftAuthority73658.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73665

namespace LeftBound73669
def owner : Owner := ⟨.program ⟨214⟩, ⟨10474⟩⟩
def transferEvent : Nat := 73669
def frameStart : Nat := 73636
def rule : BoundRule := .identity (.predecessor 0 73668 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73668 .coefficient)
      LeftBound73665.bound (LeftBound73665.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73667RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73665.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73665.derived selector witness)

def rawBound : CoeffClass := LeftBound73665.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73665.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound73665.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound73669

namespace LeftBound73686
def owner : Owner := ⟨.program ⟨214⟩, ⟨10572⟩⟩
def transferEvent : Nat := 73686
def frameStart : Nat := 73636
def rule : BoundRule := .sum [.predecessor 0 73684 .coefficient, .predecessor 1 73685 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73684 .coefficient)
      LeftBound73669.bound (LeftBound73669.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound73669.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73685 .coefficient)
      LeftAuthority73682.bound (LeftAuthority73682.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority73682.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73669.bound, LeftAuthority73682.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73669.bound, LeftAuthority73682.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73669.actual selector witness, LeftAuthority73682.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73686

namespace LeftBound73689
def owner : Owner := ⟨.program ⟨214⟩, ⟨10573⟩⟩
def transferEvent : Nat := 73689
def frameStart : Nat := 73636
def rule : BoundRule := .identity (.predecessor 0 73688 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73688 .coefficient)
      LeftBound73686.bound (LeftBound73686.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound73686.derived selector witness)

def rawBound : CoeffClass := LeftBound73686.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73686.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound73686.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound73689

namespace LeftBound73695
def owner : Owner := ⟨.program ⟨214⟩, ⟨10574⟩⟩
def transferEvent : Nat := 73695
def frameStart : Nat := 73636
def rule : BoundRule := .product (.predecessor 0 73693 .coefficient) (.predecessor 1 73694 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73693 .coefficient)
      LeftAuthority73691.bound (LeftAuthority73691.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73692RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73691.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73691.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73694 .coefficient)
      LeftBound73689.bound (LeftBound73689.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73690RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73689.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73689.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority73691.bound LeftBound73689.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73691.bound, LeftBound73689.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority73691.actual selector witness) * (LeftBound73689.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73695

namespace LeftBound73711
def owner : Owner := ⟨.program ⟨214⟩, ⟨7832⟩⟩
def transferEvent : Nat := 73711
def frameStart : Nat := 73636
def rule : BoundRule := .scale (.predecessor 0 73709 .coefficient) (.value (.predecessor 1 73710 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73709 .coefficient)
      LeftAuthority73707.bound (LeftAuthority73707.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73708RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73707.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73707.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73710 .coefficient)
      LeftAuthority73698.bound (LeftAuthority73698.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority73698.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority73707.bound LeftAuthority73698.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73707.bound, LeftAuthority73698.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority73707.actual selector witness) * (LeftAuthority73698.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound73711

namespace LeftBound73714
def owner : Owner := ⟨.program ⟨214⟩, ⟨6771⟩⟩
def transferEvent : Nat := 73714
def frameStart : Nat := 73636
def rule : BoundRule := .identity (.predecessor 0 73713 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73713 .coefficient)
      LeftAuthority73701.bound (LeftAuthority73701.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73702RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73701.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73701.derived selector witness)

def rawBound : CoeffClass := LeftAuthority73701.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73701.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority73701.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound73714

namespace LeftBound73718
def owner : Owner := ⟨.program ⟨214⟩, ⟨7833⟩⟩
def transferEvent : Nat := 73718
def frameStart : Nat := 73636
def rule : BoundRule := .product (.predecessor 0 73716 .coefficient) (.predecessor 1 73717 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73716 .coefficient)
      LeftBound73714.bound (LeftBound73714.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73715RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73714.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73714.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73717 .coefficient)
      LeftBound73711.bound (LeftBound73711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73712RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73711.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73711.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound73714.bound LeftBound73711.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73714.bound, LeftBound73711.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound73714.actual selector witness) * (LeftBound73711.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73718

namespace LeftBound73723
def owner : Owner := ⟨.program ⟨214⟩, ⟨10575⟩⟩
def transferEvent : Nat := 73723
def frameStart : Nat := 73636
def rule : BoundRule := .sum [.predecessor 0 73721 .coefficient, .predecessor 1 73722 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73721 .coefficient)
      LeftBound73718.bound (LeftBound73718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73720RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73718.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73718.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73722 .coefficient)
      LeftBound73695.bound (LeftBound73695.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73697RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73695.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73695.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73718.bound, LeftBound73695.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73718.bound, LeftBound73695.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73718.actual selector witness, LeftBound73695.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73723

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
