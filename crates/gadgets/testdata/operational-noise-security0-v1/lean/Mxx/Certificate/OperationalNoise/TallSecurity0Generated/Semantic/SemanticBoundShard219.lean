import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard174
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard218

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound33749
def owner : Owner := ⟨.program ⟨214⟩, ⟨16072⟩⟩
def transferEvent : Nat := 33749
def frameStart : Nat := 33710
def rule : BoundRule := .identity (.predecessor 0 33748 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33748 .coefficient)
      LeftAuthority33746.bound (LeftAuthority33746.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33747RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33746.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33746.derived selector witness)

def rawBound : CoeffClass := LeftAuthority33746.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33746.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority33746.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound33749

namespace LeftBound33766
def owner : Owner := ⟨.program ⟨214⟩, ⟨16146⟩⟩
def transferEvent : Nat := 33766
def frameStart : Nat := 33710
def rule : BoundRule := .sum [.predecessor 0 33764 .coefficient, .predecessor 1 33765 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33764 .coefficient)
      LeftBound33749.bound (LeftBound33749.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound33749.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33765 .coefficient)
      LeftAuthority33762.bound (LeftAuthority33762.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority33762.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound33749.bound, LeftAuthority33762.bound]
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33749.bound, LeftAuthority33762.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound33749.actual selector witness, LeftAuthority33762.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33766

namespace LeftBound33769
def owner : Owner := ⟨.program ⟨214⟩, ⟨16147⟩⟩
def transferEvent : Nat := 33769
def frameStart : Nat := 33710
def rule : BoundRule := .identity (.predecessor 0 33768 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33768 .coefficient)
      LeftBound33766.bound (LeftBound33766.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound33766.derived selector witness)

def rawBound : CoeffClass := LeftBound33766.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33766.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound33766.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound33769

namespace LeftBound33775
def owner : Owner := ⟨.program ⟨214⟩, ⟨16148⟩⟩
def transferEvent : Nat := 33775
def frameStart : Nat := 33710
def rule : BoundRule := .product (.predecessor 0 33773 .coefficient) (.predecessor 1 33774 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33773 .coefficient)
      LeftAuthority33771.bound (LeftAuthority33771.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33772RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33771.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33771.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33774 .coefficient)
      LeftBound33769.bound (LeftBound33769.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33770RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33769.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33769.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority33771.bound LeftBound33769.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33771.bound, LeftBound33769.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority33771.actual selector witness) * (LeftBound33769.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33775

namespace LeftBound33783
def owner : Owner := ⟨.program ⟨214⟩, ⟨16149⟩⟩
def transferEvent : Nat := 33783
def frameStart : Nat := 33710
def rule : BoundRule := .sum [.predecessor 0 33781 .coefficient, .predecessor 1 33782 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33781 .coefficient)
      LeftAuthority33779.bound (LeftAuthority33779.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33779.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33779.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33782 .coefficient)
      LeftBound33775.bound (LeftBound33775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33777RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33775.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33775.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority33779.bound, LeftBound33775.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33779.bound, LeftBound33775.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority33779.actual selector witness, LeftBound33775.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33783

namespace LeftBound33787
def owner : Owner := ⟨.program ⟨214⟩, ⟨28116⟩⟩
def transferEvent : Nat := 33787
def frameStart : Nat := 33710
def rule : BoundRule := .product (.predecessor 0 33785 .coefficient) (.predecessor 1 33786 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33785 .coefficient)
      LeftBound33783.bound (LeftBound33783.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33784RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33783.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33783.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33786 .coefficient)
      LeftAuthority33760.bound (LeftAuthority33760.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33761RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33760.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33760.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound33783.bound LeftAuthority33760.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33783.bound, LeftAuthority33760.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound33783.actual selector witness) * (LeftAuthority33760.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33787

namespace LeftBound33798
def owner : Owner := ⟨.program ⟨214⟩, ⟨18061⟩⟩
def transferEvent : Nat := 33798
def frameStart : Nat := 33710
def rule : BoundRule := .product (.predecessor 0 33796 .coefficient) (.predecessor 1 33797 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33796 .coefficient)
      LeftAuthority33771.bound (LeftAuthority33771.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33772RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33771.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33771.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33797 .coefficient)
      LeftAuthority33794.bound (LeftAuthority33794.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact33795RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33794.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33794.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority33771.bound LeftAuthority33794.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33771.bound, LeftAuthority33794.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority33771.actual selector witness) * (LeftAuthority33794.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33798

namespace LeftBound33806
def owner : Owner := ⟨.program ⟨214⟩, ⟨18062⟩⟩
def transferEvent : Nat := 33806
def frameStart : Nat := 33710
def rule : BoundRule := .sum [.predecessor 0 33804 .coefficient, .predecessor 1 33805 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33804 .coefficient)
      LeftAuthority33802.bound (LeftAuthority33802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact33803RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33802.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33802.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33805 .coefficient)
      LeftBound33798.bound (LeftBound33798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact33800RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33798.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33798.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority33802.bound, LeftBound33798.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33802.bound, LeftBound33798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority33802.actual selector witness, LeftBound33798.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33806

namespace LeftBound33810
def owner : Owner := ⟨.program ⟨214⟩, ⟨28121⟩⟩
def transferEvent : Nat := 33810
def frameStart : Nat := 33710
def rule : BoundRule := .sum [.predecessor 0 33808 .coefficient, .predecessor 1 33809 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33808 .coefficient)
      LeftBound33806.bound (LeftBound33806.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact33807RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33806.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33806.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33809 .coefficient)
      LeftBound33787.bound (LeftBound33787.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact33792RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33787.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33787.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound33806.bound, LeftBound33787.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33806.bound, LeftBound33787.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound33806.actual selector witness, LeftBound33787.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33810

namespace LeftBound33823
def owner : Owner := ⟨.program ⟨214⟩, ⟨28118⟩⟩
def transferEvent : Nat := 33823
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 33821 .coefficient, .predecessor 1 33822 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33821 .coefficient)
      LeftBound33652.bound (LeftBound33652.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact33820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33652.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33652.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33822 .coefficient)
      LeftBound33635.bound (LeftBound33635.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events131.exact33642RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33635.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33635.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound33652.bound, LeftBound33635.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33652.bound, LeftBound33635.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound33652.actual selector witness, LeftBound33635.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33823

namespace LeftBound33826
def owner : Owner := ⟨.program ⟨214⟩, ⟨28118⟩⟩
def transferEvent : Nat := 33826
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 33820 .summary, .result 33642 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 33820 .summary)
      LeftBound33654.bound (LeftBound33654.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21487⟩⟩) (rawTerms := some (Proof.Events132.exact33820RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound33654.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 33642 .summary)
      LeftBound33637.bound (LeftBound33637.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28117⟩⟩) (rawTerms := some (Proof.Events131.exact33642RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound33637.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound33654.bound, LeftBound33637.bound]
def bound : CoeffClass := .finite ⟨1292113298829627502592, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33654.bound, LeftBound33637.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound33654.actual selector witness, LeftBound33637.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound33826

namespace LeftBound33830
def owner : Owner := ⟨.program ⟨214⟩, ⟨28119⟩⟩
def transferEvent : Nat := 33830
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 33828 .coefficient) (.predecessor 1 33829 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33828 .coefficient)
      LeftBound33823.bound (LeftBound33823.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact33827RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound33823.bound, RecordedBoundRefines] <;> decide)
      (LeftBound33823.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33829 .coefficient)
      LeftBound5698.bound (LeftBound5698.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5699RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5698.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5698.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound33823.bound LeftBound5698.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33823.bound, LeftBound5698.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound33823.actual selector witness) * (LeftBound5698.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33830

namespace LeftBound33831
def owner : Owner := ⟨.program ⟨214⟩, ⟨28119⟩⟩
def transferEvent : Nat := 33831
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩ [⟨.result 5695 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5695 .coefficient)
      LeftAuthority5694.bound (LeftAuthority5694.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6637⟩⟩) (rawTerms := some (Proof.Events022.exact5695RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5694.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5694.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5694.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5694.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5694.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound33831

namespace LeftBound33832
def owner : Owner := ⟨.program ⟨214⟩, ⟨28119⟩⟩
def transferEvent : Nat := 33832
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 33827 .summary) (.transfer 33831) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 33827 .summary)
      LeftBound33826.bound (LeftBound33826.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28118⟩⟩) (rawTerms := some (Proof.Events132.exact33827RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound33826.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 33831)
      LeftBound33831.bound (LeftBound33831.actual selector witness) := by
  exact .transfer (LeftBound33831.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound33826.bound LeftBound33831.bound
def bound : CoeffClass := .finite ⟨4742076480517514208552681472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound33826.bound, LeftBound33831.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound33826.actual selector witness) * (LeftBound33831.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33832

namespace LeftBound33847
def owner : Owner := ⟨.program ⟨214⟩, ⟨27900⟩⟩
def transferEvent : Nat := 33847
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 33845 .coefficient) (.predecessor 1 33846 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 33845 .coefficient)
      LeftBound26514.bound (LeftBound26514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events103.exact26518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26514.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 33846 .coefficient)
      LeftAuthority33843.bound (LeftAuthority33843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events132.exact33844RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33843.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33843.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound26514.bound LeftAuthority33843.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26514.bound, LeftAuthority33843.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound26514.actual selector witness) * (LeftAuthority33843.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound33847

namespace LeftBound33848
def owner : Owner := ⟨.program ⟨214⟩, ⟨27900⟩⟩
def transferEvent : Nat := 33848
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27898⟩⟩]⟩ [⟨.result 33844 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 33844 .coefficient)
      LeftAuthority33843.bound (LeftAuthority33843.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27898⟩⟩) (rawTerms := some (Proof.Events132.exact33844RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33843.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33843.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority33843.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority33843.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority33843.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound33848

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
