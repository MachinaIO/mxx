import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard475
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard522

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound77512
def owner : Owner := ⟨.program ⟨214⟩, ⟨28065⟩⟩
def transferEvent : Nat := 77512
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 69911 .summary) (.transfer 77511) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 69911 .summary)
      LeftBound69910.bound (LeftBound69910.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26140⟩⟩) (rawTerms := some (Proof.Events273.exact69911RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69910.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 77511)
      LeftBound77511.bound (LeftBound77511.actual selector witness) := by
  exact .transfer (LeftBound77511.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound69910.bound LeftBound77511.bound
def bound : CoeffClass := .finite ⟨1292113297018323992576, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69910.bound, LeftBound77511.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound69910.actual selector witness) * (LeftBound77511.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77512

namespace LeftBound77523
def owner : Owner := ⟨.program ⟨214⟩, ⟨21470⟩⟩
def transferEvent : Nat := 77523
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 77521 .coefficient) (.value (.predecessor 1 77522 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77521 .coefficient)
      LeftAuthority77519.bound (LeftAuthority77519.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77519.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77519.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77522 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority77519.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77519.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority77519.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound77523

namespace LeftBound77527
def owner : Owner := ⟨.program ⟨214⟩, ⟨21471⟩⟩
def transferEvent : Nat := 77527
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 77525 .coefficient) (.predecessor 1 77526 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77525 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77526 .coefficient)
      LeftBound77523.bound (LeftBound77523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77524RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77523.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77523.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound77523.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound77523.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound77523.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77527

namespace LeftBound77528
def owner : Owner := ⟨.program ⟨214⟩, ⟨21471⟩⟩
def transferEvent : Nat := 77528
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21468⟩⟩]⟩ [⟨.result 77520 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77520 .coefficient)
      LeftAuthority77519.bound (LeftAuthority77519.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21468⟩⟩) (rawTerms := some (Proof.Events302.exact77520RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77519.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77519.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority77519.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77519.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority77519.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound77528

namespace LeftBound77529
def owner : Owner := ⟨.program ⟨214⟩, ⟨21471⟩⟩
def transferEvent : Nat := 77529
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 77528) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 77528)
      LeftBound77528.bound (LeftBound77528.actual selector witness) := by
  exact .transfer (LeftBound77528.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound77528.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound77528.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound77528.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77529

namespace LeftBound77624
def owner : Owner := ⟨.program ⟨214⟩, ⟨16056⟩⟩
def transferEvent : Nat := 77624
def frameStart : Nat := 77585
def rule : BoundRule := .identity (.predecessor 0 77623 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77623 .coefficient)
      LeftAuthority77621.bound (LeftAuthority77621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77622RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77621.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77621.derived selector witness)

def rawBound : CoeffClass := LeftAuthority77621.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77621.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority77621.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound77624

namespace LeftBound77641
def owner : Owner := ⟨.program ⟨214⟩, ⟨16130⟩⟩
def transferEvent : Nat := 77641
def frameStart : Nat := 77585
def rule : BoundRule := .sum [.predecessor 0 77639 .coefficient, .predecessor 1 77640 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77639 .coefficient)
      LeftBound77624.bound (LeftBound77624.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound77624.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77640 .coefficient)
      LeftAuthority77637.bound (LeftAuthority77637.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority77637.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77624.bound, LeftAuthority77637.bound]
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77624.bound, LeftAuthority77637.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound77624.actual selector witness, LeftAuthority77637.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77641

namespace LeftBound77644
def owner : Owner := ⟨.program ⟨214⟩, ⟨16131⟩⟩
def transferEvent : Nat := 77644
def frameStart : Nat := 77585
def rule : BoundRule := .identity (.predecessor 0 77643 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77643 .coefficient)
      LeftBound77641.bound (LeftBound77641.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound77641.derived selector witness)

def rawBound : CoeffClass := LeftBound77641.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77641.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound77641.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound77644

namespace LeftBound77650
def owner : Owner := ⟨.program ⟨214⟩, ⟨16132⟩⟩
def transferEvent : Nat := 77650
def frameStart : Nat := 77585
def rule : BoundRule := .product (.predecessor 0 77648 .coefficient) (.predecessor 1 77649 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77648 .coefficient)
      LeftAuthority77646.bound (LeftAuthority77646.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77647RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77646.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77646.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77649 .coefficient)
      LeftBound77644.bound (LeftBound77644.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77645RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77644.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77644.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority77646.bound LeftBound77644.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77646.bound, LeftBound77644.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority77646.actual selector witness) * (LeftBound77644.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77650

namespace LeftBound77658
def owner : Owner := ⟨.program ⟨214⟩, ⟨16133⟩⟩
def transferEvent : Nat := 77658
def frameStart : Nat := 77585
def rule : BoundRule := .sum [.predecessor 0 77656 .coefficient, .predecessor 1 77657 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77656 .coefficient)
      LeftAuthority77654.bound (LeftAuthority77654.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77654.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77654.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77657 .coefficient)
      LeftBound77650.bound (LeftBound77650.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77652RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77650.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77650.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority77654.bound, LeftBound77650.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77654.bound, LeftBound77650.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority77654.actual selector witness, LeftBound77650.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77658

namespace LeftBound77662
def owner : Owner := ⟨.program ⟨214⟩, ⟨28064⟩⟩
def transferEvent : Nat := 77662
def frameStart : Nat := 77585
def rule : BoundRule := .product (.predecessor 0 77660 .coefficient) (.predecessor 1 77661 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77660 .coefficient)
      LeftBound77658.bound (LeftBound77658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77658.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77661 .coefficient)
      LeftAuthority77635.bound (LeftAuthority77635.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77635.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77635.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound77658.bound LeftAuthority77635.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77658.bound, LeftAuthority77635.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound77658.actual selector witness) * (LeftAuthority77635.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77662

namespace LeftBound77673
def owner : Owner := ⟨.program ⟨214⟩, ⟨18033⟩⟩
def transferEvent : Nat := 77673
def frameStart : Nat := 77585
def rule : BoundRule := .product (.predecessor 0 77671 .coefficient) (.predecessor 1 77672 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77671 .coefficient)
      LeftAuthority77646.bound (LeftAuthority77646.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77647RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77646.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77646.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77672 .coefficient)
      LeftAuthority77669.bound (LeftAuthority77669.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77669.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77669.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority77646.bound LeftAuthority77669.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77646.bound, LeftAuthority77669.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority77646.actual selector witness) * (LeftAuthority77669.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77673

namespace LeftBound77681
def owner : Owner := ⟨.program ⟨214⟩, ⟨18034⟩⟩
def transferEvent : Nat := 77681
def frameStart : Nat := 77585
def rule : BoundRule := .sum [.predecessor 0 77679 .coefficient, .predecessor 1 77680 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77679 .coefficient)
      LeftAuthority77677.bound (LeftAuthority77677.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77678RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77677.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77677.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77680 .coefficient)
      LeftBound77673.bound (LeftBound77673.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77675RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77673.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77673.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority77677.bound, LeftBound77673.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77677.bound, LeftBound77673.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority77677.actual selector witness, LeftBound77673.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77681

namespace LeftBound77685
def owner : Owner := ⟨.program ⟨214⟩, ⟨28069⟩⟩
def transferEvent : Nat := 77685
def frameStart : Nat := 77585
def rule : BoundRule := .sum [.predecessor 0 77683 .coefficient, .predecessor 1 77684 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77683 .coefficient)
      LeftBound77681.bound (LeftBound77681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77682RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77681.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77681.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77684 .coefficient)
      LeftBound77662.bound (LeftBound77662.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77667RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77662.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77662.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77681.bound, LeftBound77662.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77681.bound, LeftBound77662.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound77681.actual selector witness, LeftBound77662.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77685

namespace LeftBound77698
def owner : Owner := ⟨.program ⟨214⟩, ⟨28066⟩⟩
def transferEvent : Nat := 77698
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 77696 .coefficient, .predecessor 1 77697 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 77696 .coefficient)
      LeftBound77527.bound (LeftBound77527.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77695RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77527.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77527.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 77697 .coefficient)
      LeftBound77510.bound (LeftBound77510.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77510.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77510.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77527.bound, LeftBound77510.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77527.bound, LeftBound77510.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound77527.actual selector witness, LeftBound77510.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77698

namespace LeftBound77701
def owner : Owner := ⟨.program ⟨214⟩, ⟨28066⟩⟩
def transferEvent : Nat := 77701
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 77695 .summary, .result 77517 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77695 .summary)
      LeftBound77529.bound (LeftBound77529.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21471⟩⟩) (rawTerms := some (Proof.Events303.exact77695RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77529.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 77517 .summary)
      LeftBound77512.bound (LeftBound77512.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28065⟩⟩) (rawTerms := some (Proof.Events302.exact77517RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77512.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77529.bound, LeftBound77512.bound]
def bound : CoeffClass := .finite ⟨1292113298829627502592, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77529.bound, LeftBound77512.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound77529.actual selector witness, LeftBound77512.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77701

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
