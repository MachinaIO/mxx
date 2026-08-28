import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard271

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound40612
def owner : Owner := ⟨.program ⟨214⟩, ⟨7857⟩⟩
def transferEvent : Nat := 40612
def frameStart : Nat := 40530
def rule : BoundRule := .product (.predecessor 0 40610 .coefficient) (.predecessor 1 40611 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40610 .coefficient)
      LeftBound40608.bound (LeftBound40608.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40609RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40608.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40608.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40611 .coefficient)
      LeftBound40605.bound (LeftBound40605.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40606RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40605.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40605.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40608.bound LeftBound40605.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40608.bound, LeftBound40605.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40608.actual selector witness) * (LeftBound40605.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40612

namespace LeftBound40617
def owner : Owner := ⟨.program ⟨214⟩, ⟨14542⟩⟩
def transferEvent : Nat := 40617
def frameStart : Nat := 40530
def rule : BoundRule := .sum [.predecessor 0 40615 .coefficient, .predecessor 1 40616 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40615 .coefficient)
      LeftBound40612.bound (LeftBound40612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40614RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40612.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40612.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40616 .coefficient)
      LeftBound40589.bound (LeftBound40589.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40591RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40589.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40589.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40612.bound, LeftBound40589.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40612.bound, LeftBound40589.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40612.actual selector witness, LeftBound40589.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40617

namespace LeftBound40621
def owner : Owner := ⟨.program ⟨214⟩, ⟨26156⟩⟩
def transferEvent : Nat := 40621
def frameStart : Nat := 40530
def rule : BoundRule := .product (.predecessor 0 40619 .coefficient) (.predecessor 1 40620 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40619 .coefficient)
      LeftBound40617.bound (LeftBound40617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40618RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40617.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40617.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40620 .coefficient)
      LeftAuthority40574.bound (LeftAuthority40574.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40575RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40574.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40574.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40617.bound LeftAuthority40574.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40617.bound, LeftAuthority40574.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40617.actual selector witness) * (LeftAuthority40574.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40621

namespace LeftBound40632
def owner : Owner := ⟨.program ⟨214⟩, ⟨16069⟩⟩
def transferEvent : Nat := 40632
def frameStart : Nat := 40530
def rule : BoundRule := .product (.predecessor 0 40630 .coefficient) (.predecessor 1 40631 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40630 .coefficient)
      LeftAuthority40585.bound (LeftAuthority40585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40585.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40585.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40631 .coefficient)
      LeftAuthority40628.bound (LeftAuthority40628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40629RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40628.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40628.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority40585.bound LeftAuthority40628.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40585.bound, LeftAuthority40628.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority40585.actual selector witness) * (LeftAuthority40628.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40632

namespace LeftBound40640
def owner : Owner := ⟨.program ⟨214⟩, ⟨16070⟩⟩
def transferEvent : Nat := 40640
def frameStart : Nat := 40530
def rule : BoundRule := .sum [.predecessor 0 40638 .coefficient, .predecessor 1 40639 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40638 .coefficient)
      LeftAuthority40636.bound (LeftAuthority40636.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40637RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40636.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40636.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40639 .coefficient)
      LeftBound40632.bound (LeftBound40632.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40634RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40632.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40632.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority40636.bound, LeftBound40632.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40636.bound, LeftBound40632.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority40636.actual selector witness, LeftBound40632.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40640

namespace LeftBound40644
def owner : Owner := ⟨.program ⟨214⟩, ⟨26157⟩⟩
def transferEvent : Nat := 40644
def frameStart : Nat := 40530
def rule : BoundRule := .sum [.predecessor 0 40642 .coefficient, .predecessor 1 40643 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40642 .coefficient)
      LeftBound40640.bound (LeftBound40640.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40641RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40640.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40640.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40643 .coefficient)
      LeftBound40621.bound (LeftBound40621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40626RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40621.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40621.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40640.bound, LeftBound40621.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40640.bound, LeftBound40621.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40640.actual selector witness, LeftBound40621.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40644

namespace LeftBound40657
def owner : Owner := ⟨.program ⟨214⟩, ⟨26155⟩⟩
def transferEvent : Nat := 40657
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40655 .coefficient, .predecessor 1 40656 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40655 .coefficient)
      LeftBound40478.bound (LeftBound40478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40654RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40478.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40478.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40656 .coefficient)
      LeftBound40461.bound (LeftBound40461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40461.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40461.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40478.bound, LeftBound40461.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40478.bound, LeftBound40461.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40478.actual selector witness, LeftBound40461.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40657

namespace LeftBound40660
def owner : Owner := ⟨.program ⟨214⟩, ⟨26155⟩⟩
def transferEvent : Nat := 40660
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 40654 .summary, .result 40468 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40654 .summary)
      LeftBound40480.bound (LeftBound40480.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19611⟩⟩) (rawTerms := some (Proof.Events158.exact40654RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40480.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40468 .summary)
      LeftBound40463.bound (LeftBound40463.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26154⟩⟩) (rawTerms := some (Proof.Events158.exact40468RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40463.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40480.bound, LeftBound40463.bound]
def bound : CoeffClass := .finite ⟨352072932929536, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40480.bound, LeftBound40463.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40480.actual selector witness, LeftBound40463.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40660

namespace LeftBound40664
def owner : Owner := ⟨.program ⟨214⟩, ⟨28111⟩⟩
def transferEvent : Nat := 40664
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 40662 .coefficient) (.predecessor 1 40663 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40662 .coefficient)
      LeftBound40657.bound (LeftBound40657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40657.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40657.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40663 .coefficient)
      LeftAuthority40383.bound (LeftAuthority40383.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40384RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40383.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40383.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40657.bound LeftAuthority40383.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40657.bound, LeftAuthority40383.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40657.actual selector witness) * (LeftAuthority40383.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40664

namespace LeftBound40665
def owner : Owner := ⟨.program ⟨214⟩, ⟨28111⟩⟩
def transferEvent : Nat := 40665
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28109⟩⟩]⟩ [⟨.result 40384 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40384 .coefficient)
      LeftAuthority40383.bound (LeftAuthority40383.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28109⟩⟩) (rawTerms := some (Proof.Events157.exact40384RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40383.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40383.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority40383.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40383.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority40383.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound40665

namespace LeftBound40666
def owner : Owner := ⟨.program ⟨214⟩, ⟨28111⟩⟩
def transferEvent : Nat := 40666
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 40661 .summary) (.transfer 40665) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40661 .summary)
      LeftBound40660.bound (LeftBound40660.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26155⟩⟩) (rawTerms := some (Proof.Events158.exact40661RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40660.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 40665)
      LeftBound40665.bound (LeftBound40665.actual selector witness) := by
  exact .transfer (LeftBound40665.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40660.bound LeftBound40665.bound
def bound : CoeffClass := .finite ⟨1292113297018323992576, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40660.bound, LeftBound40665.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40660.actual selector witness) * (LeftBound40665.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40666

namespace LeftBound40677
def owner : Owner := ⟨.program ⟨214⟩, ⟨21554⟩⟩
def transferEvent : Nat := 40677
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 40675 .coefficient) (.value (.predecessor 1 40676 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40675 .coefficient)
      LeftAuthority40673.bound (LeftAuthority40673.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40674RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40673.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40673.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40676 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority40673.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40673.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority40673.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound40677

namespace LeftBound40681
def owner : Owner := ⟨.program ⟨214⟩, ⟨21555⟩⟩
def transferEvent : Nat := 40681
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 40679 .coefficient) (.predecessor 1 40680 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40679 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40680 .coefficient)
      LeftBound40677.bound (LeftBound40677.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40678RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40677.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40677.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound40677.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound40677.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound40677.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40681

namespace LeftBound40682
def owner : Owner := ⟨.program ⟨214⟩, ⟨21555⟩⟩
def transferEvent : Nat := 40682
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21552⟩⟩]⟩ [⟨.result 40674 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40674 .coefficient)
      LeftAuthority40673.bound (LeftAuthority40673.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21552⟩⟩) (rawTerms := some (Proof.Events158.exact40674RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40673.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40673.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority40673.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40673.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority40673.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound40682

namespace LeftBound40683
def owner : Owner := ⟨.program ⟨214⟩, ⟨21555⟩⟩
def transferEvent : Nat := 40683
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 40682) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 40682)
      LeftBound40682.bound (LeftBound40682.actual selector witness) := by
  exact .transfer (LeftBound40682.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound40682.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound40682.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound40682.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40683

namespace LeftBound40778
def owner : Owner := ⟨.program ⟨214⟩, ⟨16068⟩⟩
def transferEvent : Nat := 40778
def frameStart : Nat := 40739
def rule : BoundRule := .identity (.predecessor 0 40777 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40777 .coefficient)
      LeftAuthority40775.bound (LeftAuthority40775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events159.exact40776RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40775.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority40775.derived selector witness)

def rawBound : CoeffClass := LeftAuthority40775.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority40775.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority40775.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound40778

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
