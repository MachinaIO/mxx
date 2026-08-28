import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard455
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard456

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound67480
def owner : Owner := ⟨.program ⟨214⟩, ⟨16548⟩⟩
def transferEvent : Nat := 67480
def frameStart : Nat := 67370
def rule : BoundRule := .sum [.predecessor 0 67478 .coefficient, .predecessor 1 67479 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67478 .coefficient)
      LeftAuthority67476.bound (LeftAuthority67476.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67477RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67476.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67476.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67479 .coefficient)
      LeftBound67472.bound (LeftBound67472.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67472.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67472.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority67476.bound, LeftBound67472.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67476.bound, LeftBound67472.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority67476.actual selector witness, LeftBound67472.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67480

namespace LeftBound67484
def owner : Owner := ⟨.program ⟨214⟩, ⟨25449⟩⟩
def transferEvent : Nat := 67484
def frameStart : Nat := 67370
def rule : BoundRule := .sum [.predecessor 0 67482 .coefficient, .predecessor 1 67483 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67482 .coefficient)
      LeftBound67480.bound (LeftBound67480.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67481RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67480.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67480.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67483 .coefficient)
      LeftBound67461.bound (LeftBound67461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67466RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67461.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67461.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67480.bound, LeftBound67461.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67480.bound, LeftBound67461.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67480.actual selector witness, LeftBound67461.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67484

namespace LeftBound67497
def owner : Owner := ⟨.program ⟨214⟩, ⟨25447⟩⟩
def transferEvent : Nat := 67497
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 67495 .coefficient, .predecessor 1 67496 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67495 .coefficient)
      LeftBound67318.bound (LeftBound67318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67496 .coefficient)
      LeftBound67301.bound (LeftBound67301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67308RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67301.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67301.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67318.bound, LeftBound67301.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67318.bound, LeftBound67301.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67318.actual selector witness, LeftBound67301.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67497

namespace LeftBound67500
def owner : Owner := ⟨.program ⟨214⟩, ⟨25447⟩⟩
def transferEvent : Nat := 67500
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 67494 .summary, .result 67308 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67494 .summary)
      LeftBound67320.bound (LeftBound67320.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19959⟩⟩) (rawTerms := some (Proof.Events263.exact67494RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67320.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67308 .summary)
      LeftBound67303.bound (LeftBound67303.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25446⟩⟩) (rawTerms := some (Proof.Events262.exact67308RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67303.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67320.bound, LeftBound67303.bound]
def bound : CoeffClass := .finite ⟨352134001995776, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67320.bound, LeftBound67303.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67320.actual selector witness, LeftBound67303.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67500

namespace LeftBound67504
def owner : Owner := ⟨.program ⟨214⟩, ⟨29157⟩⟩
def transferEvent : Nat := 67504
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 67502 .coefficient) (.predecessor 1 67503 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67502 .coefficient)
      LeftBound67497.bound (LeftBound67497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67501RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67497.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67497.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67503 .coefficient)
      LeftAuthority67223.bound (LeftAuthority67223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67224RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67223.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67223.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound67497.bound LeftAuthority67223.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67497.bound, LeftAuthority67223.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound67497.actual selector witness) * (LeftAuthority67223.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67504

namespace LeftBound67505
def owner : Owner := ⟨.program ⟨214⟩, ⟨29157⟩⟩
def transferEvent : Nat := 67505
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29155⟩⟩]⟩ [⟨.result 67224 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67224 .coefficient)
      LeftAuthority67223.bound (LeftAuthority67223.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29155⟩⟩) (rawTerms := some (Proof.Events262.exact67224RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67223.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67223.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority67223.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67223.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority67223.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound67505

namespace LeftBound67506
def owner : Owner := ⟨.program ⟨214⟩, ⟨29157⟩⟩
def transferEvent : Nat := 67506
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 67501 .summary) (.transfer 67505) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67501 .summary)
      LeftBound67500.bound (LeftBound67500.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25447⟩⟩) (rawTerms := some (Proof.Events263.exact67501RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67500.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 67505)
      LeftBound67505.bound (LeftBound67505.actual selector witness) := by
  exact .transfer (LeftBound67505.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound67500.bound LeftBound67505.bound
def bound : CoeffClass := .finite ⟨1292337421468529852416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67500.bound, LeftBound67505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound67500.actual selector witness) * (LeftBound67505.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67506

namespace LeftBound67517
def owner : Owner := ⟨.program ⟨214⟩, ⟨22262⟩⟩
def transferEvent : Nat := 67517
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 67515 .coefficient) (.value (.predecessor 1 67516 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67515 .coefficient)
      LeftAuthority67513.bound (LeftAuthority67513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67514RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67513.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67513.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67516 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority67513.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67513.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority67513.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound67517

namespace LeftBound67521
def owner : Owner := ⟨.program ⟨214⟩, ⟨22263⟩⟩
def transferEvent : Nat := 67521
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 67519 .coefficient) (.predecessor 1 67520 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67519 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67520 .coefficient)
      LeftBound67517.bound (LeftBound67517.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67517.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67517.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound67517.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound67517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound67517.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67521

namespace LeftBound67522
def owner : Owner := ⟨.program ⟨214⟩, ⟨22263⟩⟩
def transferEvent : Nat := 67522
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22260⟩⟩]⟩ [⟨.result 67514 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67514 .coefficient)
      LeftAuthority67513.bound (LeftAuthority67513.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22260⟩⟩) (rawTerms := some (Proof.Events263.exact67514RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67513.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67513.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority67513.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67513.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority67513.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound67522

namespace LeftBound67523
def owner : Owner := ⟨.program ⟨214⟩, ⟨22263⟩⟩
def transferEvent : Nat := 67523
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 67522) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 67522)
      LeftBound67522.bound (LeftBound67522.actual selector witness) := by
  exact .transfer (LeftBound67522.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound67522.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound67522.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound67522.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67523

namespace LeftBound67618
def owner : Owner := ⟨.program ⟨214⟩, ⟨16546⟩⟩
def transferEvent : Nat := 67618
def frameStart : Nat := 67579
def rule : BoundRule := .identity (.predecessor 0 67617 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67617 .coefficient)
      LeftAuthority67615.bound (LeftAuthority67615.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67615.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67615.derived selector witness)

def rawBound : CoeffClass := LeftAuthority67615.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67615.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority67615.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound67618

namespace LeftBound67635
def owner : Owner := ⟨.program ⟨214⟩, ⟨16585⟩⟩
def transferEvent : Nat := 67635
def frameStart : Nat := 67579
def rule : BoundRule := .sum [.predecessor 0 67633 .coefficient, .predecessor 1 67634 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67633 .coefficient)
      LeftBound67618.bound (LeftBound67618.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound67618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67634 .coefficient)
      LeftAuthority67631.bound (LeftAuthority67631.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority67631.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67618.bound, LeftAuthority67631.bound]
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67618.bound, LeftAuthority67631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67618.actual selector witness, LeftAuthority67631.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67635

namespace LeftBound67638
def owner : Owner := ⟨.program ⟨214⟩, ⟨16586⟩⟩
def transferEvent : Nat := 67638
def frameStart : Nat := 67579
def rule : BoundRule := .identity (.predecessor 0 67637 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67637 .coefficient)
      LeftBound67635.bound (LeftBound67635.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound67635.derived selector witness)

def rawBound : CoeffClass := LeftBound67635.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67635.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound67635.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound67638

namespace LeftBound67644
def owner : Owner := ⟨.program ⟨214⟩, ⟨16587⟩⟩
def transferEvent : Nat := 67644
def frameStart : Nat := 67579
def rule : BoundRule := .product (.predecessor 0 67642 .coefficient) (.predecessor 1 67643 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67642 .coefficient)
      LeftAuthority67640.bound (LeftAuthority67640.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67641RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67640.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67640.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67643 .coefficient)
      LeftBound67638.bound (LeftBound67638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67639RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67638.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67638.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority67640.bound LeftBound67638.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67640.bound, LeftBound67638.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority67640.actual selector witness) * (LeftBound67638.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67644

namespace LeftBound67652
def owner : Owner := ⟨.program ⟨214⟩, ⟨16588⟩⟩
def transferEvent : Nat := 67652
def frameStart : Nat := 67579
def rule : BoundRule := .sum [.predecessor 0 67650 .coefficient, .predecessor 1 67651 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67650 .coefficient)
      LeftAuthority67648.bound (LeftAuthority67648.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67649RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67648.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67648.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67651 .coefficient)
      LeftBound67644.bound (LeftBound67644.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events264.exact67646RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67644.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67644.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority67648.bound, LeftBound67644.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67648.bound, LeftBound67644.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority67648.actual selector witness, LeftBound67644.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67652

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
