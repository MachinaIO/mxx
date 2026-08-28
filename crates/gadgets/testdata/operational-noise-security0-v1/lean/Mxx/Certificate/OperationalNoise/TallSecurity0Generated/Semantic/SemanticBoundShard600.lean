import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard092
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard599

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound87634
def owner : Owner := ⟨.program ⟨214⟩, ⟨10682⟩⟩
def transferEvent : Nat := 87634
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 87632 .coefficient) (.predecessor 1 87633 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87632 .coefficient)
      LeftBound87628.bound (LeftBound87628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87631RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87628.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87628.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87633 .coefficient)
      LeftAuthority4198.bound (LeftAuthority4198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4199RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4198.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4198.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound87628.bound LeftAuthority4198.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87628.bound, LeftAuthority4198.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound87628.actual selector witness) * (LeftAuthority4198.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87634

namespace LeftBound87635
def owner : Owner := ⟨.program ⟨214⟩, ⟨10682⟩⟩
def transferEvent : Nat := 87635
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9505⟩⟩], []⟩ [⟨.result 4199 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4199 .coefficient)
      LeftAuthority4198.bound (LeftAuthority4198.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9505⟩⟩) (rawTerms := some (Proof.Events016.exact4199RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4198.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4198.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4198.bound []
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4198.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority4198.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound87635

namespace LeftBound87636
def owner : Owner := ⟨.program ⟨214⟩, ⟨10682⟩⟩
def transferEvent : Nat := 87636
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 87631 .summary) (.transfer 87635) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87631 .summary)
      LeftBound87629.bound (LeftBound87629.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10681⟩⟩) (rawTerms := some (Proof.Events342.exact87631RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87629.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 87635)
      LeftBound87635.bound (LeftBound87635.actual selector witness) := by
  exact .transfer (LeftBound87635.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound87629.bound LeftBound87635.bound
def bound : CoeffClass := .finite ⟨2496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87629.bound, LeftBound87635.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound87629.actual selector witness) * (LeftBound87635.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87636

namespace LeftBound87642
def owner : Owner := ⟨.program ⟨214⟩, ⟨9506⟩⟩
def transferEvent : Nat := 87642
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 87640 .coefficient) (.predecessor 1 87641 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87640 .coefficient)
      LeftAuthority4198.bound (LeftAuthority4198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4199RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4198.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4198.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87641 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4198.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4198.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4198.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound87642

namespace LeftBound87647
def owner : Owner := ⟨.program ⟨214⟩, ⟨7238⟩⟩
def transferEvent : Nat := 87647
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 87645 .coefficient) (.predecessor 1 87646 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87645 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87646 .coefficient)
      LeftBound14528.bound (LeftBound14528.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14529RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14528.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14528.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound14528.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound14528.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound14528.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87647

namespace LeftBound87652
def owner : Owner := ⟨.program ⟨214⟩, ⟨9507⟩⟩
def transferEvent : Nat := 87652
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 87650 .coefficient, .predecessor 1 87651 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87650 .coefficient)
      LeftBound87647.bound (LeftBound87647.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87649RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87647.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87647.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87651 .coefficient)
      LeftBound87642.bound (LeftBound87642.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87644RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87642.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87642.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87647.bound, LeftBound87642.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87647.bound, LeftBound87642.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87647.actual selector witness, LeftBound87642.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87652

namespace LeftBound87656
def owner : Owner := ⟨.program ⟨214⟩, ⟨9508⟩⟩
def transferEvent : Nat := 87656
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 87654 .coefficient, .predecessor 1 87655 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87654 .coefficient)
      LeftBound87652.bound (LeftBound87652.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87652.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87652.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87655 .coefficient)
      LeftBound14520.bound (LeftBound14520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14520.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87652.bound, LeftBound14520.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87652.bound, LeftBound14520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87652.actual selector witness, LeftBound14520.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87656

namespace LeftBound87657
def owner : Owner := ⟨.program ⟨214⟩, ⟨9508⟩⟩
def transferEvent : Nat := 87657
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨96⟩⟩]⟩ [⟨.result 14521 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14521 .coefficient)
      LeftBound14520.bound (LeftBound14520.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨96⟩⟩) (rawTerms := some (Proof.Events056.exact14521RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14520.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound14520.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound14520.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound87657

namespace LeftBound87662
def owner : Owner := ⟨.program ⟨214⟩, ⟨9509⟩⟩
def transferEvent : Nat := 87662
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 87660 .coefficient) (.predecessor 1 87661 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87660 .coefficient)
      LeftBound87656.bound (LeftBound87656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87656.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87656.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87661 .coefficient)
      LeftBound14517.bound (LeftBound14517.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14517.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14517.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound87656.bound LeftBound14517.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87656.bound, LeftBound14517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound87656.actual selector witness) * (LeftBound14517.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87662

namespace LeftBound87663
def owner : Owner := ⟨.program ⟨214⟩, ⟨9509⟩⟩
def transferEvent : Nat := 87663
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩ [⟨.result 14514 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14514 .coefficient)
      LeftAuthority14513.bound (LeftAuthority14513.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7834⟩⟩) (rawTerms := some (Proof.Events056.exact14514RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14513.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14513.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority14513.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14513.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority14513.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound87663

namespace LeftBound87664
def owner : Owner := ⟨.program ⟨214⟩, ⟨9509⟩⟩
def transferEvent : Nat := 87664
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 87659 .summary) (.transfer 87663) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87659 .summary)
      LeftBound87657.bound (LeftBound87657.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9508⟩⟩) (rawTerms := some (Proof.Events342.exact87659RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87657.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 87663)
      LeftBound87663.bound (LeftBound87663.actual selector witness) := by
  exact .transfer (LeftBound87663.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound87657.bound LeftBound87663.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87657.bound, LeftBound87663.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound87657.actual selector witness) * (LeftBound87663.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87664

namespace LeftBound87672
def owner : Owner := ⟨.program ⟨214⟩, ⟨10683⟩⟩
def transferEvent : Nat := 87672
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 87670 .coefficient, .predecessor 1 87671 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87670 .coefficient)
      LeftBound87662.bound (LeftBound87662.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87669RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87662.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87662.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87671 .coefficient)
      LeftBound87634.bound (LeftBound87634.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87639RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87634.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87634.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87662.bound, LeftBound87634.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87662.bound, LeftBound87634.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87662.actual selector witness, LeftBound87634.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87672

namespace LeftBound87674
def owner : Owner := ⟨.program ⟨214⟩, ⟨10683⟩⟩
def transferEvent : Nat := 87674
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 87669 .summary, .result 87639 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87669 .summary)
      LeftBound87664.bound (LeftBound87664.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9509⟩⟩) (rawTerms := some (Proof.Events342.exact87669RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87664.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87639 .summary)
      LeftBound87636.bound (LeftBound87636.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10682⟩⟩) (rawTerms := some (Proof.Events342.exact87639RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87636.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87664.bound, LeftBound87636.bound]
def bound : CoeffClass := .finite ⟨95422912, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87664.bound, LeftBound87636.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87664.actual selector witness, LeftBound87636.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87674

namespace LeftBound87678
def owner : Owner := ⟨.program ⟨214⟩, ⟨24989⟩⟩
def transferEvent : Nat := 87678
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 87676 .coefficient) (.predecessor 1 87677 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87676 .coefficient)
      LeftBound87672.bound (LeftBound87672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87675RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87672.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87672.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87677 .coefficient)
      LeftAuthority87610.bound (LeftAuthority87610.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87610.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87610.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound87672.bound LeftAuthority87610.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87672.bound, LeftAuthority87610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound87672.actual selector witness) * (LeftAuthority87610.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87678

namespace LeftBound87679
def owner : Owner := ⟨.program ⟨214⟩, ⟨24989⟩⟩
def transferEvent : Nat := 87679
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨24988⟩⟩]⟩ [⟨.result 87611 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87611 .coefficient)
      LeftAuthority87610.bound (LeftAuthority87610.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨24988⟩⟩) (rawTerms := some (Proof.Events342.exact87611RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87610.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87610.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority87610.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority87610.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound87679

namespace LeftBound87680
def owner : Owner := ⟨.program ⟨214⟩, ⟨24989⟩⟩
def transferEvent : Nat := 87680
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 87675 .summary) (.transfer 87679) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 87675 .summary)
      LeftBound87674.bound (LeftBound87674.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10683⟩⟩) (rawTerms := some (Proof.Events342.exact87675RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87674.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 87679)
      LeftBound87679.bound (LeftBound87679.actual selector witness) := by
  exact .transfer (LeftBound87679.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound87674.bound LeftBound87679.bound
def bound : CoeffClass := .finite ⟨350203613806592, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87674.bound, LeftBound87679.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound87674.actual selector witness) * (LeftBound87679.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87680

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
