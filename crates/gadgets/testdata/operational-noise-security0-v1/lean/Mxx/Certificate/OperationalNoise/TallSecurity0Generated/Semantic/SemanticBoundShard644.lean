import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard032
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard033
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard643

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound94832
def owner : Owner := ⟨.program ⟨214⟩, ⟨13135⟩⟩
def transferEvent : Nat := 94832
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94830 .coefficient, .predecessor 1 94831 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94830 .coefficient)
      LeftBound94828.bound (LeftBound94828.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94829RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94828.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94828.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94831 .coefficient)
      LeftBound6964.bound (LeftBound6964.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6965RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6964.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6964.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94828.bound, LeftBound6964.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94828.bound, LeftBound6964.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94828.actual selector witness, LeftBound6964.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94832

namespace LeftBound94833
def owner : Owner := ⟨.program ⟨214⟩, ⟨13135⟩⟩
def transferEvent : Nat := 94833
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨103⟩⟩]⟩ [⟨.result 6965 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6965 .coefficient)
      LeftBound6964.bound (LeftBound6964.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨103⟩⟩) (rawTerms := some (Proof.Events027.exact6965RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6964.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6964.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound6964.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6964.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound6964.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound94833

namespace LeftBound94838
def owner : Owner := ⟨.program ⟨214⟩, ⟨13136⟩⟩
def transferEvent : Nat := 94838
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 94836 .coefficient) (.predecessor 1 94837 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94836 .coefficient)
      LeftBound94832.bound (LeftBound94832.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94835RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94832.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94832.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94837 .coefficient)
      LeftAuthority4591.bound (LeftAuthority4591.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4591.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4591.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound94832.bound LeftAuthority4591.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94832.bound, LeftAuthority4591.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound94832.actual selector witness) * (LeftAuthority4591.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94838

namespace LeftBound94839
def owner : Owner := ⟨.program ⟨214⟩, ⟨13136⟩⟩
def transferEvent : Nat := 94839
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨10225⟩⟩], []⟩ [⟨.result 4592 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4592 .coefficient)
      LeftAuthority4591.bound (LeftAuthority4591.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨10225⟩⟩) (rawTerms := some (Proof.Events017.exact4592RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4591.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4591.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4591.bound []
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4591.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority4591.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound94839

namespace LeftBound94840
def owner : Owner := ⟨.program ⟨214⟩, ⟨13136⟩⟩
def transferEvent : Nat := 94840
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94835 .summary) (.transfer 94839) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94835 .summary)
      LeftBound94833.bound (LeftBound94833.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13135⟩⟩) (rawTerms := some (Proof.Events370.exact94835RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94833.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 94839)
      LeftBound94839.bound (LeftBound94839.actual selector witness) := by
  exact .transfer (LeftBound94839.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound94833.bound LeftBound94839.bound
def bound : CoeffClass := .finite ⟨48256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94833.bound, LeftBound94839.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound94833.actual selector witness) * (LeftBound94839.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94840

namespace LeftBound94846
def owner : Owner := ⟨.program ⟨214⟩, ⟨10226⟩⟩
def transferEvent : Nat := 94846
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 94844 .coefficient) (.predecessor 1 94845 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94844 .coefficient)
      LeftAuthority4591.bound (LeftAuthority4591.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4591.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4591.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94845 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4591.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4591.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4591.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound94846

namespace LeftBound94851
def owner : Owner := ⟨.program ⟨214⟩, ⟨7106⟩⟩
def transferEvent : Nat := 94851
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 94849 .coefficient) (.predecessor 1 94850 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94849 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94850 .coefficient)
      LeftBound7013.bound (LeftBound7013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7014RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7013.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound7013.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound7013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound7013.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94851

namespace LeftBound94856
def owner : Owner := ⟨.program ⟨214⟩, ⟨10227⟩⟩
def transferEvent : Nat := 94856
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94854 .coefficient, .predecessor 1 94855 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94854 .coefficient)
      LeftBound94851.bound (LeftBound94851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94853RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94851.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94851.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94855 .coefficient)
      LeftBound94846.bound (LeftBound94846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94848RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94846.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94846.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94851.bound, LeftBound94846.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94851.bound, LeftBound94846.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94851.actual selector witness, LeftBound94846.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94856

namespace LeftBound94860
def owner : Owner := ⟨.program ⟨214⟩, ⟨10228⟩⟩
def transferEvent : Nat := 94860
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94858 .coefficient, .predecessor 1 94859 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94858 .coefficient)
      LeftBound94856.bound (LeftBound94856.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94857RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94856.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94856.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94859 .coefficient)
      LeftBound7005.bound (LeftBound7005.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7006RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7005.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94856.bound, LeftBound7005.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94856.bound, LeftBound7005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94856.actual selector witness, LeftBound7005.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94860

namespace LeftBound94861
def owner : Owner := ⟨.program ⟨214⟩, ⟨10228⟩⟩
def transferEvent : Nat := 94861
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨83⟩⟩]⟩ [⟨.result 7006 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7006 .coefficient)
      LeftBound7005.bound (LeftBound7005.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨83⟩⟩) (rawTerms := some (Proof.Events027.exact7006RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7005.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound7005.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound7005.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound94861

namespace LeftBound94866
def owner : Owner := ⟨.program ⟨214⟩, ⟨10229⟩⟩
def transferEvent : Nat := 94866
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 94864 .coefficient) (.predecessor 1 94865 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94864 .coefficient)
      LeftBound94860.bound (LeftBound94860.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94863RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94860.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94860.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94865 .coefficient)
      LeftBound7002.bound (LeftBound7002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7002.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7002.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound94860.bound LeftBound7002.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94860.bound, LeftBound7002.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound94860.actual selector witness) * (LeftBound7002.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94866

namespace LeftBound94867
def owner : Owner := ⟨.program ⟨214⟩, ⟨10229⟩⟩
def transferEvent : Nat := 94867
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩ [⟨.result 6999 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6999 .coefficient)
      LeftAuthority6998.bound (LeftAuthority6998.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7879⟩⟩) (rawTerms := some (Proof.Events027.exact6999RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6998.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6998.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6998.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6998.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6998.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound94867

namespace LeftBound94868
def owner : Owner := ⟨.program ⟨214⟩, ⟨10229⟩⟩
def transferEvent : Nat := 94868
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94863 .summary) (.transfer 94867) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94863 .summary)
      LeftBound94861.bound (LeftBound94861.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10228⟩⟩) (rawTerms := some (Proof.Events370.exact94863RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94861.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 94867)
      LeftBound94867.bound (LeftBound94867.actual selector witness) := by
  exact .transfer (LeftBound94867.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound94861.bound LeftBound94867.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94861.bound, LeftBound94867.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound94861.actual selector witness) * (LeftBound94867.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94868

namespace LeftBound94876
def owner : Owner := ⟨.program ⟨214⟩, ⟨13137⟩⟩
def transferEvent : Nat := 94876
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94874 .coefficient, .predecessor 1 94875 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94874 .coefficient)
      LeftBound94866.bound (LeftBound94866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94866.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94866.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94875 .coefficient)
      LeftBound94838.bound (LeftBound94838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94843RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94838.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94838.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94866.bound, LeftBound94838.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94866.bound, LeftBound94838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94866.actual selector witness, LeftBound94838.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94876

namespace LeftBound94878
def owner : Owner := ⟨.program ⟨214⟩, ⟨13137⟩⟩
def transferEvent : Nat := 94878
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94873 .summary, .result 94843 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94873 .summary)
      LeftBound94868.bound (LeftBound94868.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10229⟩⟩) (rawTerms := some (Proof.Events370.exact94873RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94843 .summary)
      LeftBound94840.bound (LeftBound94840.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13136⟩⟩) (rawTerms := some (Proof.Events370.exact94843RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94840.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94868.bound, LeftBound94840.bound]
def bound : CoeffClass := .finite ⟨95468672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94868.bound, LeftBound94840.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94868.actual selector witness, LeftBound94840.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94878

namespace LeftBound94882
def owner : Owner := ⟨.program ⟨214⟩, ⟨25669⟩⟩
def transferEvent : Nat := 94882
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 94880 .coefficient) (.predecessor 1 94881 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94880 .coefficient)
      LeftBound94876.bound (LeftBound94876.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94879RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94876.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94876.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94881 .coefficient)
      LeftAuthority94814.bound (LeftAuthority94814.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94814.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94814.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound94876.bound LeftAuthority94814.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94876.bound, LeftAuthority94814.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound94876.actual selector witness) * (LeftAuthority94814.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94882

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
