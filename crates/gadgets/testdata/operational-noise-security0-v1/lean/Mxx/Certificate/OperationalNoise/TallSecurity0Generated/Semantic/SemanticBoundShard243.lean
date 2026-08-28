import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard242

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound36756
def owner : Owner := ⟨.program ⟨214⟩, ⟨7881⟩⟩
def transferEvent : Nat := 36756
def frameStart : Nat := 36674
def rule : BoundRule := .product (.predecessor 0 36754 .coefficient) (.predecessor 1 36755 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36754 .coefficient)
      LeftBound36752.bound (LeftBound36752.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36753RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36752.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36752.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36755 .coefficient)
      LeftBound36749.bound (LeftBound36749.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36749.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36749.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36752.bound LeftBound36749.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36752.bound, LeftBound36749.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36752.actual selector witness) * (LeftBound36749.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36756

namespace LeftBound36761
def owner : Owner := ⟨.program ⟨214⟩, ⟨13261⟩⟩
def transferEvent : Nat := 36761
def frameStart : Nat := 36674
def rule : BoundRule := .sum [.predecessor 0 36759 .coefficient, .predecessor 1 36760 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36759 .coefficient)
      LeftBound36756.bound (LeftBound36756.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36758RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36756.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36756.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36760 .coefficient)
      LeftBound36733.bound (LeftBound36733.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36735RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36733.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36733.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36756.bound, LeftBound36733.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36756.bound, LeftBound36733.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36756.actual selector witness, LeftBound36733.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36761

namespace LeftBound36765
def owner : Owner := ⟨.program ⟨214⟩, ⟨25694⟩⟩
def transferEvent : Nat := 36765
def frameStart : Nat := 36674
def rule : BoundRule := .product (.predecessor 0 36763 .coefficient) (.predecessor 1 36764 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36763 .coefficient)
      LeftBound36761.bound (LeftBound36761.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36761.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36761.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36764 .coefficient)
      LeftAuthority36718.bound (LeftAuthority36718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36718.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36718.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36761.bound LeftAuthority36718.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36761.bound, LeftAuthority36718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36761.actual selector witness) * (LeftAuthority36718.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36765

namespace LeftBound36776
def owner : Owner := ⟨.program ⟨214⟩, ⟨16881⟩⟩
def transferEvent : Nat := 36776
def frameStart : Nat := 36674
def rule : BoundRule := .product (.predecessor 0 36774 .coefficient) (.predecessor 1 36775 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36774 .coefficient)
      LeftAuthority36729.bound (LeftAuthority36729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36730RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36729.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36775 .coefficient)
      LeftAuthority36772.bound (LeftAuthority36772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36773RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36772.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36772.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority36729.bound LeftAuthority36772.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36729.bound, LeftAuthority36772.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority36729.actual selector witness) * (LeftAuthority36772.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36776

namespace LeftBound36784
def owner : Owner := ⟨.program ⟨214⟩, ⟨16882⟩⟩
def transferEvent : Nat := 36784
def frameStart : Nat := 36674
def rule : BoundRule := .sum [.predecessor 0 36782 .coefficient, .predecessor 1 36783 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36782 .coefficient)
      LeftAuthority36780.bound (LeftAuthority36780.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36781RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36780.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36780.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36783 .coefficient)
      LeftBound36776.bound (LeftBound36776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36776.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority36780.bound, LeftBound36776.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36780.bound, LeftBound36776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority36780.actual selector witness, LeftBound36776.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36784

namespace LeftBound36788
def owner : Owner := ⟨.program ⟨214⟩, ⟨25695⟩⟩
def transferEvent : Nat := 36788
def frameStart : Nat := 36674
def rule : BoundRule := .sum [.predecessor 0 36786 .coefficient, .predecessor 1 36787 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36786 .coefficient)
      LeftBound36784.bound (LeftBound36784.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36785RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36784.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36784.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36787 .coefficient)
      LeftBound36765.bound (LeftBound36765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36770RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36765.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36765.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36784.bound, LeftBound36765.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36784.bound, LeftBound36765.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36784.actual selector witness, LeftBound36765.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36788

namespace LeftBound36801
def owner : Owner := ⟨.program ⟨214⟩, ⟨25693⟩⟩
def transferEvent : Nat := 36801
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 36799 .coefficient, .predecessor 1 36800 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36799 .coefficient)
      LeftBound36622.bound (LeftBound36622.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36798RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36622.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36622.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36800 .coefficient)
      LeftBound36605.bound (LeftBound36605.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36612RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36605.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36605.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36622.bound, LeftBound36605.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36622.bound, LeftBound36605.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36622.actual selector witness, LeftBound36605.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36801

namespace LeftBound36804
def owner : Owner := ⟨.program ⟨214⟩, ⟨25693⟩⟩
def transferEvent : Nat := 36804
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 36798 .summary, .result 36612 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36798 .summary)
      LeftBound36624.bound (LeftBound36624.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20187⟩⟩) (rawTerms := some (Proof.Events143.exact36798RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36624.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36612 .summary)
      LeftBound36607.bound (LeftBound36607.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25692⟩⟩) (rawTerms := some (Proof.Events143.exact36612RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36607.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36624.bound, LeftBound36607.bound]
def bound : CoeffClass := .finite ⟨352182857248768, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36624.bound, LeftBound36607.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36624.actual selector witness, LeftBound36607.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36804

namespace LeftBound36808
def owner : Owner := ⟨.program ⟨214⟩, ⟨29847⟩⟩
def transferEvent : Nat := 36808
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 36806 .coefficient) (.predecessor 1 36807 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36806 .coefficient)
      LeftBound36801.bound (LeftBound36801.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36805RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36801.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36801.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36807 .coefficient)
      LeftAuthority36527.bound (LeftAuthority36527.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events142.exact36528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36527.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36527.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36801.bound LeftAuthority36527.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36801.bound, LeftAuthority36527.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36801.actual selector witness) * (LeftAuthority36527.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36808

namespace LeftBound36809
def owner : Owner := ⟨.program ⟨214⟩, ⟨29847⟩⟩
def transferEvent : Nat := 36809
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29845⟩⟩]⟩ [⟨.result 36528 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36528 .coefficient)
      LeftAuthority36527.bound (LeftAuthority36527.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29845⟩⟩) (rawTerms := some (Proof.Events142.exact36528RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36527.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36527.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority36527.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36527.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority36527.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound36809

namespace LeftBound36810
def owner : Owner := ⟨.program ⟨214⟩, ⟨29847⟩⟩
def transferEvent : Nat := 36810
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36805 .summary) (.transfer 36809) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36805 .summary)
      LeftBound36804.bound (LeftBound36804.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25693⟩⟩) (rawTerms := some (Proof.Events143.exact36805RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36804.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 36809)
      LeftBound36809.bound (LeftBound36809.actual selector witness) := by
  exact .transfer (LeftBound36809.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36804.bound LeftBound36809.bound
def bound : CoeffClass := .finite ⟨1292516721028694540288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36804.bound, LeftBound36809.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36804.actual selector witness) * (LeftBound36809.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36810

namespace LeftBound36821
def owner : Owner := ⟨.program ⟨214⟩, ⟨22706⟩⟩
def transferEvent : Nat := 36821
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 36819 .coefficient) (.value (.predecessor 1 36820 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36819 .coefficient)
      LeftAuthority36817.bound (LeftAuthority36817.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36818RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36817.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36817.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36820 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority36817.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36817.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority36817.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound36821

namespace LeftBound36825
def owner : Owner := ⟨.program ⟨214⟩, ⟨22707⟩⟩
def transferEvent : Nat := 36825
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 36823 .coefficient) (.predecessor 1 36824 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36823 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36824 .coefficient)
      LeftBound36821.bound (LeftBound36821.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36822RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36821.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36821.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound36821.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound36821.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound36821.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36825

namespace LeftBound36826
def owner : Owner := ⟨.program ⟨214⟩, ⟨22707⟩⟩
def transferEvent : Nat := 36826
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22704⟩⟩]⟩ [⟨.result 36818 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36818 .coefficient)
      LeftAuthority36817.bound (LeftAuthority36817.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22704⟩⟩) (rawTerms := some (Proof.Events143.exact36818RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36817.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36817.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority36817.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36817.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority36817.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound36826

namespace LeftBound36827
def owner : Owner := ⟨.program ⟨214⟩, ⟨22707⟩⟩
def transferEvent : Nat := 36827
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 36826) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 36826)
      LeftBound36826.bound (LeftBound36826.actual selector witness) := by
  exact .transfer (LeftBound36826.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound36826.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound36826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound36826.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36827

namespace LeftBound36922
def owner : Owner := ⟨.program ⟨214⟩, ⟨16880⟩⟩
def transferEvent : Nat := 36922
def frameStart : Nat := 36883
def rule : BoundRule := .identity (.predecessor 0 36921 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36921 .coefficient)
      LeftAuthority36919.bound (LeftAuthority36919.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36919.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36919.derived selector witness)

def rawBound : CoeffClass := LeftAuthority36919.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36919.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority36919.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound36922

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
