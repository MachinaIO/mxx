import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard342
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard346
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard405

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound59416
def owner : Owner := ⟨.program ⟨214⟩, ⟨29836⟩⟩
def transferEvent : Nat := 59416
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 59412 .summary, .result 51625 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59412 .summary)
      LeftBound59411.bound (LeftBound59411.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29619⟩⟩) (rawTerms := some (Proof.Events232.exact59412RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59411.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51625 .summary)
      LeftBound51624.bound (LeftBound51624.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29835⟩⟩) (rawTerms := some (Proof.Events201.exact51625RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51624.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59411.bound, LeftBound51624.bound]
def bound : CoeffClass := .finite ⟨21966497597451692486656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59411.bound, LeftBound51624.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59411.actual selector witness, LeftBound51624.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59416

namespace LeftBound59420
def owner : Owner := ⟨.program ⟨214⟩, ⟨30143⟩⟩
def transferEvent : Nat := 59420
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 59418 .coefficient, .predecessor 1 59419 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59418 .coefficient)
      LeftBound59415.bound (LeftBound59415.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59417RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59415.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59415.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59419 .coefficient)
      LeftBound51139.bound (LeftBound51139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51143RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51139.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51139.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59415.bound, LeftBound51139.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59415.bound, LeftBound51139.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59415.actual selector witness, LeftBound51139.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59420

namespace LeftBound59421
def owner : Owner := ⟨.program ⟨214⟩, ⟨30143⟩⟩
def transferEvent : Nat := 59421
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 59417 .summary, .result 51143 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59417 .summary)
      LeftBound59416.bound (LeftBound59416.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29836⟩⟩) (rawTerms := some (Proof.Events232.exact59417RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59416.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51143 .summary)
      LeftBound51142.bound (LeftBound51142.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30142⟩⟩) (rawTerms := some (Proof.Events199.exact51143RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51142.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59416.bound, LeftBound51142.bound]
def bound : CoeffClass := .finite ⟨23259036732736711122944, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59416.bound, LeftBound51142.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound59416.actual selector witness, LeftBound51142.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59421

namespace LeftBound59425
def owner : Owner := ⟨.program ⟨214⟩, ⟨30144⟩⟩
def transferEvent : Nat := 59425
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 59423 .coefficient) (.predecessor 1 59424 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59423 .coefficient)
      LeftBound59420.bound (LeftBound59420.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59422RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59420.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59420.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59424 .coefficient)
      LeftAuthority50644.bound (LeftAuthority50644.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50645RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50644.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50644.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound59420.bound LeftAuthority50644.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59420.bound, LeftAuthority50644.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound59420.actual selector witness) * (LeftAuthority50644.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59425

namespace LeftBound59426
def owner : Owner := ⟨.program ⟨214⟩, ⟨30144⟩⟩
def transferEvent : Nat := 59426
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩ [⟨.result 50645 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50645 .coefficient)
      LeftAuthority50644.bound (LeftAuthority50644.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18684⟩⟩) (rawTerms := some (Proof.Events197.exact50645RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50644.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50644.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority50644.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50644.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority50644.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound59426

namespace LeftBound59427
def owner : Owner := ⟨.program ⟨214⟩, ⟨30144⟩⟩
def transferEvent : Nat := 59427
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 59422 .summary) (.transfer 59426) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59422 .summary)
      LeftBound59421.bound (LeftBound59421.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30143⟩⟩) (rawTerms := some (Proof.Events232.exact59422RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59421.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 59426)
      LeftBound59426.bound (LeftBound59426.actual selector witness) := by
  exact .transfer (LeftBound59426.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound59421.bound LeftBound59426.bound
def bound : CoeffClass := .finite ⟨85361036953731453608582447104, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59421.bound, LeftBound59426.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound59421.actual selector witness) * (LeftBound59426.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59427

namespace LeftBound59506
def owner : Owner := ⟨.program ⟨214⟩, ⟨18565⟩⟩
def transferEvent : Nat := 59506
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 59504 .coefficient) (.value (.predecessor 1 59505 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59504 .coefficient)
      LeftAuthority59502.bound (LeftAuthority59502.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59502.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59502.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59505 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority59502.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59502.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority59502.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound59506

namespace LeftBound59510
def owner : Owner := ⟨.program ⟨214⟩, ⟨18566⟩⟩
def transferEvent : Nat := 59510
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 59508 .coefficient) (.predecessor 1 59509 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 59508 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 59509 .coefficient)
      LeftBound59506.bound (LeftBound59506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59507RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59506.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound59506.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound59506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound59506.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59510

namespace LeftBound59511
def owner : Owner := ⟨.program ⟨214⟩, ⟨18566⟩⟩
def transferEvent : Nat := 59511
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨18563⟩⟩]⟩ [⟨.result 59503 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59503 .coefficient)
      LeftAuthority59502.bound (LeftAuthority59502.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18563⟩⟩) (rawTerms := some (Proof.Events232.exact59503RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59502.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59502.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority59502.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59502.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority59502.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound59511

namespace LeftBound59512
def owner : Owner := ⟨.program ⟨214⟩, ⟨18566⟩⟩
def transferEvent : Nat := 59512
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 59511) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 59511)
      LeftBound59511.bound (LeftBound59511.actual selector witness) := by
  exact .transfer (LeftBound59511.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound59511.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound59511.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound59511.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59512

namespace LeftBound60540
def owner : Owner := ⟨.program ⟨214⟩, ⟨15315⟩⟩
def transferEvent : Nat := 60540
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60538 .coefficient, .predecessor 1 60539 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60538 .coefficient)
      LeftAuthority60536.bound (LeftAuthority60536.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60537RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60536.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60536.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60539 .coefficient)
      LeftAuthority60513.bound (LeftAuthority60513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60514RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60513.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60513.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority60536.bound, LeftAuthority60513.bound]
def bound : CoeffClass := .finite ⟨91, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority60536.bound, LeftAuthority60513.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority60536.actual selector witness, LeftAuthority60513.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60540

namespace LeftBound60544
def owner : Owner := ⟨.program ⟨214⟩, ⟨15371⟩⟩
def transferEvent : Nat := 60544
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60542 .coefficient, .predecessor 1 60543 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60542 .coefficient)
      LeftBound60540.bound (LeftBound60540.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60541RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60540.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60540.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60543 .coefficient)
      LeftAuthority60490.bound (LeftAuthority60490.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60491RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60490.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60490.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60540.bound, LeftAuthority60490.bound]
def bound : CoeffClass := .finite ⟨142, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60540.bound, LeftAuthority60490.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60540.actual selector witness, LeftAuthority60490.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60544

namespace LeftBound60548
def owner : Owner := ⟨.program ⟨214⟩, ⟨17337⟩⟩
def transferEvent : Nat := 60548
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60546 .coefficient, .predecessor 1 60547 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60546 .coefficient)
      LeftBound60544.bound (LeftBound60544.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60545RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60544.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60544.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60547 .coefficient)
      LeftAuthority60467.bound (LeftAuthority60467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60467.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60467.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60544.bound, LeftAuthority60467.bound]
def bound : CoeffClass := .finite ⟨197, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60544.bound, LeftAuthority60467.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60544.actual selector witness, LeftAuthority60467.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60548

namespace LeftBound60552
def owner : Owner := ⟨.program ⟨214⟩, ⟨17338⟩⟩
def transferEvent : Nat := 60552
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60550 .coefficient, .predecessor 1 60551 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60550 .coefficient)
      LeftBound60548.bound (LeftBound60548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60549RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60548.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60548.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60551 .coefficient)
      LeftAuthority60444.bound (LeftAuthority60444.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60445RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60444.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60444.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60548.bound, LeftAuthority60444.bound]
def bound : CoeffClass := .finite ⟨255, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60548.bound, LeftAuthority60444.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60548.actual selector witness, LeftAuthority60444.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60552

namespace LeftBound60556
def owner : Owner := ⟨.program ⟨214⟩, ⟨17339⟩⟩
def transferEvent : Nat := 60556
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60554 .coefficient, .predecessor 1 60555 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60554 .coefficient)
      LeftBound60552.bound (LeftBound60552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60552.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60552.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60555 .coefficient)
      LeftAuthority60421.bound (LeftAuthority60421.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60422RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60421.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60421.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60552.bound, LeftAuthority60421.bound]
def bound : CoeffClass := .finite ⟨314, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60552.bound, LeftAuthority60421.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60552.actual selector witness, LeftAuthority60421.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60556

namespace LeftBound60560
def owner : Owner := ⟨.program ⟨214⟩, ⟨17340⟩⟩
def transferEvent : Nat := 60560
def frameStart : Nat := 60103
def rule : BoundRule := .sum [.predecessor 0 60558 .coefficient, .predecessor 1 60559 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 60558 .coefficient)
      LeftBound60556.bound (LeftBound60556.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60557RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60556.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60556.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 60559 .coefficient)
      LeftAuthority60398.bound (LeftAuthority60398.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events235.exact60399RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority60398.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority60398.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60556.bound, LeftAuthority60398.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60556.bound, LeftAuthority60398.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound60556.actual selector witness, LeftAuthority60398.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60560

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
