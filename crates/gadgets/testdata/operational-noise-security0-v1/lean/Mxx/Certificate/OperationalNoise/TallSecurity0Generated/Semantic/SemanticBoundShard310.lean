import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard247
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard309

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound46695
def owner : Owner := ⟨.program ⟨214⟩, ⟨16975⟩⟩
def transferEvent : Nat := 46695
def frameStart : Nat := 46639
def rule : BoundRule := .sum [.predecessor 0 46693 .coefficient, .predecessor 1 46694 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46693 .coefficient)
      LeftBound46678.bound (LeftBound46678.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound46678.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46694 .coefficient)
      LeftAuthority46691.bound (LeftAuthority46691.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority46691.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46678.bound, LeftAuthority46691.bound]
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46678.bound, LeftAuthority46691.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46678.actual selector witness, LeftAuthority46691.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46695

namespace LeftBound46698
def owner : Owner := ⟨.program ⟨214⟩, ⟨16976⟩⟩
def transferEvent : Nat := 46698
def frameStart : Nat := 46639
def rule : BoundRule := .identity (.predecessor 0 46697 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46697 .coefficient)
      LeftBound46695.bound (LeftBound46695.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound46695.derived selector witness)

def rawBound : CoeffClass := LeftBound46695.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46695.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound46695.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound46698

namespace LeftBound46704
def owner : Owner := ⟨.program ⟨214⟩, ⟨16977⟩⟩
def transferEvent : Nat := 46704
def frameStart : Nat := 46639
def rule : BoundRule := .product (.predecessor 0 46702 .coefficient) (.predecessor 1 46703 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46702 .coefficient)
      LeftAuthority46700.bound (LeftAuthority46700.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46701RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46700.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46700.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46703 .coefficient)
      LeftBound46698.bound (LeftBound46698.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46699RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46698.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46698.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority46700.bound LeftBound46698.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46700.bound, LeftBound46698.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority46700.actual selector witness) * (LeftBound46698.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46704

namespace LeftBound46712
def owner : Owner := ⟨.program ⟨214⟩, ⟨16978⟩⟩
def transferEvent : Nat := 46712
def frameStart : Nat := 46639
def rule : BoundRule := .sum [.predecessor 0 46710 .coefficient, .predecessor 1 46711 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46710 .coefficient)
      LeftAuthority46708.bound (LeftAuthority46708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46709RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46708.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46708.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46711 .coefficient)
      LeftBound46704.bound (LeftBound46704.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46706RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46704.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46704.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority46708.bound, LeftBound46704.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46708.bound, LeftBound46704.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority46708.actual selector witness, LeftBound46704.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46712

namespace LeftBound46716
def owner : Owner := ⟨.program ⟨214⟩, ⟨29839⟩⟩
def transferEvent : Nat := 46716
def frameStart : Nat := 46639
def rule : BoundRule := .product (.predecessor 0 46714 .coefficient) (.predecessor 1 46715 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46714 .coefficient)
      LeftBound46712.bound (LeftBound46712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46712.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46715 .coefficient)
      LeftAuthority46689.bound (LeftAuthority46689.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46690RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46689.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46689.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound46712.bound LeftAuthority46689.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46712.bound, LeftAuthority46689.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound46712.actual selector witness) * (LeftAuthority46689.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46716

namespace LeftBound46727
def owner : Owner := ⟨.program ⟨214⟩, ⟨16937⟩⟩
def transferEvent : Nat := 46727
def frameStart : Nat := 46639
def rule : BoundRule := .product (.predecessor 0 46725 .coefficient) (.predecessor 1 46726 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46725 .coefficient)
      LeftAuthority46700.bound (LeftAuthority46700.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46701RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46700.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46700.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46726 .coefficient)
      LeftAuthority46723.bound (LeftAuthority46723.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46724RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46723.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46723.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority46700.bound LeftAuthority46723.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46700.bound, LeftAuthority46723.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority46700.actual selector witness) * (LeftAuthority46723.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46727

namespace LeftBound46735
def owner : Owner := ⟨.program ⟨214⟩, ⟨16938⟩⟩
def transferEvent : Nat := 46735
def frameStart : Nat := 46639
def rule : BoundRule := .sum [.predecessor 0 46733 .coefficient, .predecessor 1 46734 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46733 .coefficient)
      LeftAuthority46731.bound (LeftAuthority46731.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46731.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46731.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46734 .coefficient)
      LeftBound46727.bound (LeftBound46727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46729RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46727.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46727.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority46731.bound, LeftBound46727.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46731.bound, LeftBound46727.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority46731.actual selector witness, LeftBound46727.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46735

namespace LeftBound46739
def owner : Owner := ⟨.program ⟨214⟩, ⟨29844⟩⟩
def transferEvent : Nat := 46739
def frameStart : Nat := 46639
def rule : BoundRule := .sum [.predecessor 0 46737 .coefficient, .predecessor 1 46738 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46737 .coefficient)
      LeftBound46735.bound (LeftBound46735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46735.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46735.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46738 .coefficient)
      LeftBound46716.bound (LeftBound46716.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46721RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46716.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46716.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46735.bound, LeftBound46716.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46735.bound, LeftBound46716.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46735.actual selector witness, LeftBound46716.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46739

namespace LeftBound46752
def owner : Owner := ⟨.program ⟨214⟩, ⟨29841⟩⟩
def transferEvent : Nat := 46752
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 46750 .coefficient, .predecessor 1 46751 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46750 .coefficient)
      LeftBound46581.bound (LeftBound46581.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46749RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46581.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46581.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46751 .coefficient)
      LeftBound46564.bound (LeftBound46564.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46571RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46564.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46564.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46581.bound, LeftBound46564.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46581.bound, LeftBound46564.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46581.actual selector witness, LeftBound46564.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46752

namespace LeftBound46755
def owner : Owner := ⟨.program ⟨214⟩, ⟨29841⟩⟩
def transferEvent : Nat := 46755
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 46749 .summary, .result 46571 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46749 .summary)
      LeftBound46583.bound (LeftBound46583.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22635⟩⟩) (rawTerms := some (Proof.Events182.exact46749RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46583.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46571 .summary)
      LeftBound46566.bound (LeftBound46566.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29840⟩⟩) (rawTerms := some (Proof.Events181.exact46571RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46566.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46583.bound, LeftBound46566.bound]
def bound : CoeffClass := .finite ⟨1292516722839998050304, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46583.bound, LeftBound46566.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46583.actual selector witness, LeftBound46566.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46755

namespace LeftBound46759
def owner : Owner := ⟨.program ⟨214⟩, ⟨29842⟩⟩
def transferEvent : Nat := 46759
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 46757 .coefficient) (.predecessor 1 46758 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46757 .coefficient)
      LeftBound46752.bound (LeftBound46752.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46756RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46752.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46752.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46758 .coefficient)
      LeftBound5538.bound (LeftBound5538.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5539RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5538.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5538.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound46752.bound LeftBound5538.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46752.bound, LeftBound5538.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound46752.actual selector witness) * (LeftBound5538.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46759

namespace LeftBound46760
def owner : Owner := ⟨.program ⟨214⟩, ⟨29842⟩⟩
def transferEvent : Nat := 46760
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6659⟩⟩]⟩ [⟨.result 5535 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5535 .coefficient)
      LeftAuthority5534.bound (LeftAuthority5534.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6659⟩⟩) (rawTerms := some (Proof.Events021.exact5535RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5534.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5534.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5534.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5534.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound46760

namespace LeftBound46761
def owner : Owner := ⟨.program ⟨214⟩, ⟨29842⟩⟩
def transferEvent : Nat := 46761
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 46756 .summary) (.transfer 46760) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46756 .summary)
      LeftBound46755.bound (LeftBound46755.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29841⟩⟩) (rawTerms := some (Proof.Events182.exact46756RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46755.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 46760)
      LeftBound46760.bound (LeftBound46760.actual selector witness) := by
  exact .transfer (LeftBound46760.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound46755.bound LeftBound46760.bound
def bound : CoeffClass := .finite ⟨4743557053090358284584484864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46755.bound, LeftBound46760.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound46755.actual selector witness) * (LeftBound46760.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46761

namespace LeftBound46776
def owner : Owner := ⟨.program ⟨214⟩, ⟨29623⟩⟩
def transferEvent : Nat := 46776
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 46774 .coefficient) (.predecessor 1 46775 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46774 .coefficient)
      LeftBound37283.bound (LeftBound37283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37283.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37283.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46775 .coefficient)
      LeftAuthority46772.bound (LeftAuthority46772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46773RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46772.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46772.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound37283.bound LeftAuthority46772.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37283.bound, LeftAuthority46772.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound37283.actual selector witness) * (LeftAuthority46772.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46776

namespace LeftBound46777
def owner : Owner := ⟨.program ⟨214⟩, ⟨29623⟩⟩
def transferEvent : Nat := 46777
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29621⟩⟩]⟩ [⟨.result 46773 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46773 .coefficient)
      LeftAuthority46772.bound (LeftAuthority46772.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29621⟩⟩) (rawTerms := some (Proof.Events182.exact46773RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46772.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46772.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority46772.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46772.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority46772.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound46777

namespace LeftBound46778
def owner : Owner := ⟨.program ⟨214⟩, ⟨29623⟩⟩
def transferEvent : Nat := 46778
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 37287 .summary) (.transfer 46777) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37287 .summary)
      LeftBound37286.bound (LeftBound37286.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25616⟩⟩) (rawTerms := some (Proof.Events145.exact37287RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound37286.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 46777)
      LeftBound46777.bound (LeftBound46777.actual selector witness) := by
  exact .transfer (LeftBound46777.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound37286.bound LeftBound46777.bound
def bound : CoeffClass := .finite ⟨1292449483693632782336, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37286.bound, LeftBound46777.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound37286.actual selector witness) * (LeftBound46777.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46778

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
