import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard503

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound73727
def owner : Owner := ⟨.program ⟨214⟩, ⟨24909⟩⟩
def transferEvent : Nat := 73727
def frameStart : Nat := 73636
def rule : BoundRule := .product (.predecessor 0 73725 .coefficient) (.predecessor 1 73726 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73725 .coefficient)
      LeftBound73723.bound (LeftBound73723.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73724RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73723.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73723.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73726 .coefficient)
      LeftAuthority73680.bound (LeftAuthority73680.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73681RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73680.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73680.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound73723.bound LeftAuthority73680.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73723.bound, LeftAuthority73680.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound73723.actual selector witness) * (LeftAuthority73680.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73727

namespace LeftBound73738
def owner : Owner := ⟨.program ⟨214⟩, ⟨14790⟩⟩
def transferEvent : Nat := 73738
def frameStart : Nat := 73636
def rule : BoundRule := .product (.predecessor 0 73736 .coefficient) (.predecessor 1 73737 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73736 .coefficient)
      LeftAuthority73691.bound (LeftAuthority73691.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73692RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73691.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73691.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73737 .coefficient)
      LeftAuthority73734.bound (LeftAuthority73734.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73735RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73734.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73734.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority73691.bound LeftAuthority73734.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73691.bound, LeftAuthority73734.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority73691.actual selector witness) * (LeftAuthority73734.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73738

namespace LeftBound73746
def owner : Owner := ⟨.program ⟨214⟩, ⟨14791⟩⟩
def transferEvent : Nat := 73746
def frameStart : Nat := 73636
def rule : BoundRule := .sum [.predecessor 0 73744 .coefficient, .predecessor 1 73745 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73744 .coefficient)
      LeftAuthority73742.bound (LeftAuthority73742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73743RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73742.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73745 .coefficient)
      LeftBound73738.bound (LeftBound73738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73738.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73738.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority73742.bound, LeftBound73738.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73742.bound, LeftBound73738.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority73742.actual selector witness, LeftBound73738.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73746

namespace LeftBound73750
def owner : Owner := ⟨.program ⟨214⟩, ⟨24910⟩⟩
def transferEvent : Nat := 73750
def frameStart : Nat := 73636
def rule : BoundRule := .sum [.predecessor 0 73748 .coefficient, .predecessor 1 73749 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73748 .coefficient)
      LeftBound73746.bound (LeftBound73746.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73747RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73746.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73746.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73749 .coefficient)
      LeftBound73727.bound (LeftBound73727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73727.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73727.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73746.bound, LeftBound73727.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73746.bound, LeftBound73727.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73746.actual selector witness, LeftBound73727.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73750

namespace LeftBound73763
def owner : Owner := ⟨.program ⟨214⟩, ⟨24908⟩⟩
def transferEvent : Nat := 73763
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 73761 .coefficient, .predecessor 1 73762 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73761 .coefficient)
      LeftBound73584.bound (LeftBound73584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73584.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73762 .coefficient)
      LeftBound73567.bound (LeftBound73567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73574RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73567.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73567.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73584.bound, LeftBound73567.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73584.bound, LeftBound73567.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73584.actual selector witness, LeftBound73567.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73763

namespace LeftBound73766
def owner : Owner := ⟨.program ⟨214⟩, ⟨24908⟩⟩
def transferEvent : Nat := 73766
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 73760 .summary, .result 73574 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73760 .summary)
      LeftBound73586.bound (LeftBound73586.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19023⟩⟩) (rawTerms := some (Proof.Events288.exact73760RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73586.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73574 .summary)
      LeftBound73569.bound (LeftBound73569.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24907⟩⟩) (rawTerms := some (Proof.Events287.exact73574RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73569.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73586.bound, LeftBound73569.bound]
def bound : CoeffClass := .finite ⟨352011863863296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73586.bound, LeftBound73569.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73586.actual selector witness, LeftBound73569.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73766

namespace LeftBound73770
def owner : Owner := ⟨.program ⟨214⟩, ⟨26348⟩⟩
def transferEvent : Nat := 73770
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 73768 .coefficient) (.predecessor 1 73769 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73768 .coefficient)
      LeftBound73763.bound (LeftBound73763.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73763.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73763.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73769 .coefficient)
      LeftAuthority73489.bound (LeftAuthority73489.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events287.exact73490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73489.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73489.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound73763.bound LeftAuthority73489.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73763.bound, LeftAuthority73489.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound73763.actual selector witness) * (LeftAuthority73489.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73770

namespace LeftBound73771
def owner : Owner := ⟨.program ⟨214⟩, ⟨26348⟩⟩
def transferEvent : Nat := 73771
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26346⟩⟩]⟩ [⟨.result 73490 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73490 .coefficient)
      LeftAuthority73489.bound (LeftAuthority73489.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26346⟩⟩) (rawTerms := some (Proof.Events287.exact73490RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73489.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73489.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority73489.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73489.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority73489.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound73771

namespace LeftBound73772
def owner : Owner := ⟨.program ⟨214⟩, ⟨26348⟩⟩
def transferEvent : Nat := 73772
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 73767 .summary) (.transfer 73771) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73767 .summary)
      LeftBound73766.bound (LeftBound73766.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24908⟩⟩) (rawTerms := some (Proof.Events288.exact73767RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73766.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 73771)
      LeftBound73771.bound (LeftBound73771.actual selector witness) := by
  exact .transfer (LeftBound73771.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound73766.bound LeftBound73771.bound
def bound : CoeffClass := .finite ⟨1291889172568118132736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73766.bound, LeftBound73771.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound73766.actual selector witness) * (LeftBound73771.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73772

namespace LeftBound73783
def owner : Owner := ⟨.program ⟨214⟩, ⟨20390⟩⟩
def transferEvent : Nat := 73783
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 73781 .coefficient) (.value (.predecessor 1 73782 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73781 .coefficient)
      LeftAuthority73779.bound (LeftAuthority73779.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73779.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73779.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73782 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority73779.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73779.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority73779.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound73783

namespace LeftBound73787
def owner : Owner := ⟨.program ⟨214⟩, ⟨20391⟩⟩
def transferEvent : Nat := 73787
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 73785 .coefficient) (.predecessor 1 73786 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73785 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73786 .coefficient)
      LeftBound73783.bound (LeftBound73783.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73784RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73783.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73783.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound73783.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound73783.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound73783.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73787

namespace LeftBound73788
def owner : Owner := ⟨.program ⟨214⟩, ⟨20391⟩⟩
def transferEvent : Nat := 73788
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20388⟩⟩]⟩ [⟨.result 73780 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73780 .coefficient)
      LeftAuthority73779.bound (LeftAuthority73779.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20388⟩⟩) (rawTerms := some (Proof.Events288.exact73780RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73779.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73779.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority73779.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73779.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority73779.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound73788

namespace LeftBound73789
def owner : Owner := ⟨.program ⟨214⟩, ⟨20391⟩⟩
def transferEvent : Nat := 73789
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 73788) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 73788)
      LeftBound73788.bound (LeftBound73788.actual selector witness) := by
  exact .transfer (LeftBound73788.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound73788.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound73788.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound73788.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73789

namespace LeftBound73884
def owner : Owner := ⟨.program ⟨214⟩, ⟨14789⟩⟩
def transferEvent : Nat := 73884
def frameStart : Nat := 73845
def rule : BoundRule := .identity (.predecessor 0 73883 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73883 .coefficient)
      LeftAuthority73881.bound (LeftAuthority73881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events288.exact73882RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73881.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73881.derived selector witness)

def rawBound : CoeffClass := LeftAuthority73881.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority73881.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound73884

namespace LeftBound73901
def owner : Owner := ⟨.program ⟨214⟩, ⟨14828⟩⟩
def transferEvent : Nat := 73901
def frameStart : Nat := 73845
def rule : BoundRule := .sum [.predecessor 0 73899 .coefficient, .predecessor 1 73900 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73899 .coefficient)
      LeftBound73884.bound (LeftBound73884.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound73884.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 73900 .coefficient)
      LeftAuthority73897.bound (LeftAuthority73897.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority73897.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73884.bound, LeftAuthority73897.bound]
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73884.bound, LeftAuthority73897.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound73884.actual selector witness, LeftAuthority73897.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73901

namespace LeftBound73904
def owner : Owner := ⟨.program ⟨214⟩, ⟨14829⟩⟩
def transferEvent : Nat := 73904
def frameStart : Nat := 73845
def rule : BoundRule := .identity (.predecessor 0 73903 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 73903 .coefficient)
      LeftBound73901.bound (LeftBound73901.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound73901.derived selector witness)

def rawBound : CoeffClass := LeftBound73901.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound73901.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound73904

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
