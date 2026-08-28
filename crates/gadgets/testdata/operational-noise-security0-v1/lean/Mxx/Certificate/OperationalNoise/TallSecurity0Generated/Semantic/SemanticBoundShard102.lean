import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound15628
def owner : Owner := ⟨.program ⟨214⟩, ⟨18577⟩⟩
def transferEvent : Nat := 15628
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 15626 .coefficient) (.value (.predecessor 1 15627 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15626 .coefficient)
      LeftAuthority15624.bound (LeftAuthority15624.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15624.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15624.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15627 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority15624.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15624.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority15624.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound15628

namespace LeftBound15632
def owner : Owner := ⟨.program ⟨214⟩, ⟨18578⟩⟩
def transferEvent : Nat := 15632
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 15630 .coefficient) (.predecessor 1 15631 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 15630 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 15631 .coefficient)
      LeftBound15628.bound (LeftBound15628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15629RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15628.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15628.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound15628.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound15628.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound15628.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15632

namespace LeftBound15633
def owner : Owner := ⟨.program ⟨214⟩, ⟨18578⟩⟩
def transferEvent : Nat := 15633
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨18575⟩⟩]⟩ [⟨.result 15625 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15625 .coefficient)
      LeftAuthority15624.bound (LeftAuthority15624.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18575⟩⟩) (rawTerms := some (Proof.Events061.exact15625RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15624.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15624.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15624.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15624.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority15624.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound15633

namespace LeftBound15634
def owner : Owner := ⟨.program ⟨214⟩, ⟨18578⟩⟩
def transferEvent : Nat := 15634
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 15633) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 15633)
      LeftBound15633.bound (LeftBound15633.actual selector witness) := by
  exact .transfer (LeftBound15633.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound15633.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound15633.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound15633.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound15634

namespace LeftBound16662
def owner : Owner := ⟨.program ⟨214⟩, ⟨15327⟩⟩
def transferEvent : Nat := 16662
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16660 .coefficient, .predecessor 1 16661 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16660 .coefficient)
      LeftAuthority16658.bound (LeftAuthority16658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16658.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16658.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16661 .coefficient)
      LeftAuthority16635.bound (LeftAuthority16635.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events064.exact16636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16635.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16635.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority16658.bound, LeftAuthority16635.bound]
def bound : CoeffClass := .finite ⟨91, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority16658.bound, LeftAuthority16635.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority16658.actual selector witness, LeftAuthority16635.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16662

namespace LeftBound16666
def owner : Owner := ⟨.program ⟨214⟩, ⟨15383⟩⟩
def transferEvent : Nat := 16666
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16664 .coefficient, .predecessor 1 16665 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16664 .coefficient)
      LeftBound16662.bound (LeftBound16662.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16662.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16662.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16665 .coefficient)
      LeftAuthority16612.bound (LeftAuthority16612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events064.exact16613RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16612.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16612.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16662.bound, LeftAuthority16612.bound]
def bound : CoeffClass := .finite ⟨142, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16662.bound, LeftAuthority16612.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16662.actual selector witness, LeftAuthority16612.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16666

namespace LeftBound16670
def owner : Owner := ⟨.program ⟨214⟩, ⟨17364⟩⟩
def transferEvent : Nat := 16670
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16668 .coefficient, .predecessor 1 16669 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16668 .coefficient)
      LeftBound16666.bound (LeftBound16666.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16667RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16666.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16666.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16669 .coefficient)
      LeftAuthority16589.bound (LeftAuthority16589.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events064.exact16590RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16589.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16589.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16666.bound, LeftAuthority16589.bound]
def bound : CoeffClass := .finite ⟨197, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16666.bound, LeftAuthority16589.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16666.actual selector witness, LeftAuthority16589.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16670

namespace LeftBound16674
def owner : Owner := ⟨.program ⟨214⟩, ⟨17365⟩⟩
def transferEvent : Nat := 16674
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16672 .coefficient, .predecessor 1 16673 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16672 .coefficient)
      LeftBound16670.bound (LeftBound16670.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16671RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16670.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16670.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16673 .coefficient)
      LeftAuthority16566.bound (LeftAuthority16566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events064.exact16567RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16566.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16566.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16670.bound, LeftAuthority16566.bound]
def bound : CoeffClass := .finite ⟨255, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16670.bound, LeftAuthority16566.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16670.actual selector witness, LeftAuthority16566.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16674

namespace LeftBound16678
def owner : Owner := ⟨.program ⟨214⟩, ⟨17366⟩⟩
def transferEvent : Nat := 16678
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16676 .coefficient, .predecessor 1 16677 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16676 .coefficient)
      LeftBound16674.bound (LeftBound16674.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16675RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16674.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16674.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16677 .coefficient)
      LeftAuthority16543.bound (LeftAuthority16543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events064.exact16544RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16543.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16543.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16674.bound, LeftAuthority16543.bound]
def bound : CoeffClass := .finite ⟨314, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16674.bound, LeftAuthority16543.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16674.actual selector witness, LeftAuthority16543.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16678

namespace LeftBound16682
def owner : Owner := ⟨.program ⟨214⟩, ⟨17367⟩⟩
def transferEvent : Nat := 16682
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16680 .coefficient, .predecessor 1 16681 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16680 .coefficient)
      LeftBound16678.bound (LeftBound16678.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16679RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16678.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16678.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16681 .coefficient)
      LeftAuthority16520.bound (LeftAuthority16520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events064.exact16521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16520.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16520.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16678.bound, LeftAuthority16520.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16678.bound, LeftAuthority16520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16678.actual selector witness, LeftAuthority16520.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16682

namespace LeftBound16686
def owner : Owner := ⟨.program ⟨214⟩, ⟨17368⟩⟩
def transferEvent : Nat := 16686
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16684 .coefficient, .predecessor 1 16685 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16684 .coefficient)
      LeftBound16682.bound (LeftBound16682.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16682.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16682.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16685 .coefficient)
      LeftAuthority16497.bound (LeftAuthority16497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events064.exact16498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16497.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16497.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16682.bound, LeftAuthority16497.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16682.bound, LeftAuthority16497.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16682.actual selector witness, LeftAuthority16497.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16686

namespace LeftBound16690
def owner : Owner := ⟨.program ⟨214⟩, ⟨17369⟩⟩
def transferEvent : Nat := 16690
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16688 .coefficient, .predecessor 1 16689 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16688 .coefficient)
      LeftBound16686.bound (LeftBound16686.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16687RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16686.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16686.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16689 .coefficient)
      LeftAuthority16474.bound (LeftAuthority16474.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events064.exact16475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16474.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16474.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16686.bound, LeftAuthority16474.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16686.bound, LeftAuthority16474.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16686.actual selector witness, LeftAuthority16474.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16690

namespace LeftBound16694
def owner : Owner := ⟨.program ⟨214⟩, ⟨18393⟩⟩
def transferEvent : Nat := 16694
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16692 .coefficient, .predecessor 1 16693 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16692 .coefficient)
      LeftBound16690.bound (LeftBound16690.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16691RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16690.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16690.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16693 .coefficient)
      LeftAuthority16451.bound (LeftAuthority16451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events064.exact16452RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16451.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16451.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16690.bound, LeftAuthority16451.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16690.bound, LeftAuthority16451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16690.actual selector witness, LeftAuthority16451.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16694

namespace LeftBound16698
def owner : Owner := ⟨.program ⟨214⟩, ⟨18394⟩⟩
def transferEvent : Nat := 16698
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16696 .coefficient, .predecessor 1 16697 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16696 .coefficient)
      LeftBound16694.bound (LeftBound16694.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16695RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16694.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16694.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16697 .coefficient)
      LeftAuthority16428.bound (LeftAuthority16428.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events064.exact16429RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16428.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16428.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16694.bound, LeftAuthority16428.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16694.bound, LeftAuthority16428.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16694.actual selector witness, LeftAuthority16428.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16698

namespace LeftBound16702
def owner : Owner := ⟨.program ⟨214⟩, ⟨18395⟩⟩
def transferEvent : Nat := 16702
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16700 .coefficient, .predecessor 1 16701 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16700 .coefficient)
      LeftBound16698.bound (LeftBound16698.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16699RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16698.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16698.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16701 .coefficient)
      LeftAuthority16405.bound (LeftAuthority16405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events064.exact16406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16405.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16405.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16698.bound, LeftAuthority16405.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16698.bound, LeftAuthority16405.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16698.actual selector witness, LeftAuthority16405.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16702

namespace LeftBound16706
def owner : Owner := ⟨.program ⟨214⟩, ⟨18396⟩⟩
def transferEvent : Nat := 16706
def frameStart : Nat := 16225
def rule : BoundRule := .sum [.predecessor 0 16704 .coefficient, .predecessor 1 16705 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 16704 .coefficient)
      LeftBound16702.bound (LeftBound16702.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16703RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16702.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16702.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 16705 .coefficient)
      LeftAuthority16382.bound (LeftAuthority16382.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events063.exact16383RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16382.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16382.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound16702.bound, LeftAuthority16382.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound16702.bound, LeftAuthority16382.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound16702.actual selector witness, LeftAuthority16382.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound16706

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
