import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard376

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound55715
def owner : Owner := ⟨.program ⟨214⟩, ⟨6759⟩⟩
def transferEvent : Nat := 55715
def frameStart : Nat := 55637
def rule : BoundRule := .identity (.predecessor 0 55714 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55714 .coefficient)
      LeftAuthority55702.bound (LeftAuthority55702.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55703RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55702.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55702.derived selector witness)

def rawBound : CoeffClass := LeftAuthority55702.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55702.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority55702.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound55715

namespace LeftBound55719
def owner : Owner := ⟨.program ⟨214⟩, ⟨7854⟩⟩
def transferEvent : Nat := 55719
def frameStart : Nat := 55637
def rule : BoundRule := .product (.predecessor 0 55717 .coefficient) (.predecessor 1 55718 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55717 .coefficient)
      LeftBound55715.bound (LeftBound55715.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55716RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55715.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55715.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55718 .coefficient)
      LeftBound55712.bound (LeftBound55712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55712.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55712.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound55715.bound LeftBound55712.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55715.bound, LeftBound55712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound55715.actual selector witness) * (LeftBound55712.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55719

namespace LeftBound55724
def owner : Owner := ⟨.program ⟨214⟩, ⟨14321⟩⟩
def transferEvent : Nat := 55724
def frameStart : Nat := 55637
def rule : BoundRule := .sum [.predecessor 0 55722 .coefficient, .predecessor 1 55723 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55722 .coefficient)
      LeftBound55719.bound (LeftBound55719.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55721RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55719.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55719.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55723 .coefficient)
      LeftBound55696.bound (LeftBound55696.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55698RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55696.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55696.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55719.bound, LeftBound55696.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55719.bound, LeftBound55696.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55719.actual selector witness, LeftBound55696.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55724

namespace LeftBound55728
def owner : Owner := ⟨.program ⟨214⟩, ⟨26074⟩⟩
def transferEvent : Nat := 55728
def frameStart : Nat := 55637
def rule : BoundRule := .product (.predecessor 0 55726 .coefficient) (.predecessor 1 55727 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55726 .coefficient)
      LeftBound55724.bound (LeftBound55724.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55724.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55724.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55727 .coefficient)
      LeftAuthority55681.bound (LeftAuthority55681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55682RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55681.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55681.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound55724.bound LeftAuthority55681.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55724.bound, LeftAuthority55681.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound55724.actual selector witness) * (LeftAuthority55681.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55728

namespace LeftBound55739
def owner : Owner := ⟨.program ⟨214⟩, ⟨15946⟩⟩
def transferEvent : Nat := 55739
def frameStart : Nat := 55637
def rule : BoundRule := .product (.predecessor 0 55737 .coefficient) (.predecessor 1 55738 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55737 .coefficient)
      LeftAuthority55692.bound (LeftAuthority55692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55692.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55692.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55738 .coefficient)
      LeftAuthority55735.bound (LeftAuthority55735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55735.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55735.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority55692.bound LeftAuthority55735.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55692.bound, LeftAuthority55735.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority55692.actual selector witness) * (LeftAuthority55735.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55739

namespace LeftBound55747
def owner : Owner := ⟨.program ⟨214⟩, ⟨15947⟩⟩
def transferEvent : Nat := 55747
def frameStart : Nat := 55637
def rule : BoundRule := .sum [.predecessor 0 55745 .coefficient, .predecessor 1 55746 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55745 .coefficient)
      LeftAuthority55743.bound (LeftAuthority55743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55743.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55746 .coefficient)
      LeftBound55739.bound (LeftBound55739.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55741RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55739.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55739.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority55743.bound, LeftBound55739.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55743.bound, LeftBound55739.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority55743.actual selector witness, LeftBound55739.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55747

namespace LeftBound55751
def owner : Owner := ⟨.program ⟨214⟩, ⟨26075⟩⟩
def transferEvent : Nat := 55751
def frameStart : Nat := 55637
def rule : BoundRule := .sum [.predecessor 0 55749 .coefficient, .predecessor 1 55750 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55749 .coefficient)
      LeftBound55747.bound (LeftBound55747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55748RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55747.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55747.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55750 .coefficient)
      LeftBound55728.bound (LeftBound55728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55733RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55728.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55728.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55747.bound, LeftBound55728.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55747.bound, LeftBound55728.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55747.actual selector witness, LeftBound55728.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55751

namespace LeftBound55764
def owner : Owner := ⟨.program ⟨214⟩, ⟨26073⟩⟩
def transferEvent : Nat := 55764
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55762 .coefficient, .predecessor 1 55763 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55762 .coefficient)
      LeftBound55585.bound (LeftBound55585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55761RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55585.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55585.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55763 .coefficient)
      LeftBound55568.bound (LeftBound55568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55575RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55568.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55568.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55585.bound, LeftBound55568.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55585.bound, LeftBound55568.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55585.actual selector witness, LeftBound55568.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55764

namespace LeftBound55767
def owner : Owner := ⟨.program ⟨214⟩, ⟨26073⟩⟩
def transferEvent : Nat := 55767
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55761 .summary, .result 55575 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55761 .summary)
      LeftBound55587.bound (LeftBound55587.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19535⟩⟩) (rawTerms := some (Proof.Events217.exact55761RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55587.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55575 .summary)
      LeftBound55570.bound (LeftBound55570.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26072⟩⟩) (rawTerms := some (Proof.Events217.exact55575RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55570.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55587.bound, LeftBound55570.bound]
def bound : CoeffClass := .finite ⟨352060719116288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55587.bound, LeftBound55570.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55587.actual selector witness, LeftBound55570.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55767

namespace LeftBound55771
def owner : Owner := ⟨.program ⟨214⟩, ⟨27881⟩⟩
def transferEvent : Nat := 55771
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 55769 .coefficient) (.predecessor 1 55770 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55769 .coefficient)
      LeftBound55764.bound (LeftBound55764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55768RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55764.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55764.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55770 .coefficient)
      LeftAuthority55490.bound (LeftAuthority55490.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55491RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55490.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55490.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound55764.bound LeftAuthority55490.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55764.bound, LeftAuthority55490.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound55764.actual selector witness) * (LeftAuthority55490.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55771

namespace LeftBound55772
def owner : Owner := ⟨.program ⟨214⟩, ⟨27881⟩⟩
def transferEvent : Nat := 55772
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27879⟩⟩]⟩ [⟨.result 55491 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55491 .coefficient)
      LeftAuthority55490.bound (LeftAuthority55490.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27879⟩⟩) (rawTerms := some (Proof.Events216.exact55491RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55490.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55490.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority55490.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55490.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority55490.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound55772

namespace LeftBound55773
def owner : Owner := ⟨.program ⟨214⟩, ⟨27881⟩⟩
def transferEvent : Nat := 55773
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 55768 .summary) (.transfer 55772) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55768 .summary)
      LeftBound55767.bound (LeftBound55767.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26073⟩⟩) (rawTerms := some (Proof.Events217.exact55768RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55767.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 55772)
      LeftBound55772.bound (LeftBound55772.actual selector witness) := by
  exact .transfer (LeftBound55772.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound55767.bound LeftBound55772.bound
def bound : CoeffClass := .finite ⟨1292068472128282820608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55767.bound, LeftBound55772.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound55767.actual selector witness) * (LeftBound55772.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55773

namespace LeftBound55784
def owner : Owner := ⟨.program ⟨214⟩, ⟨21406⟩⟩
def transferEvent : Nat := 55784
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 55782 .coefficient) (.value (.predecessor 1 55783 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55782 .coefficient)
      LeftAuthority55780.bound (LeftAuthority55780.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55781RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55780.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55780.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55783 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority55780.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55780.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority55780.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound55784

namespace LeftBound55788
def owner : Owner := ⟨.program ⟨214⟩, ⟨21407⟩⟩
def transferEvent : Nat := 55788
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 55786 .coefficient) (.predecessor 1 55787 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55786 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55787 .coefficient)
      LeftBound55784.bound (LeftBound55784.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55785RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55784.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55784.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound55784.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound55784.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound55784.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55788

namespace LeftBound55789
def owner : Owner := ⟨.program ⟨214⟩, ⟨21407⟩⟩
def transferEvent : Nat := 55789
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21404⟩⟩]⟩ [⟨.result 55781 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55781 .coefficient)
      LeftAuthority55780.bound (LeftAuthority55780.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21404⟩⟩) (rawTerms := some (Proof.Events217.exact55781RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55780.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55780.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority55780.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55780.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority55780.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound55789

namespace LeftBound55790
def owner : Owner := ⟨.program ⟨214⟩, ⟨21407⟩⟩
def transferEvent : Nat := 55790
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 55789) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 55789)
      LeftBound55789.bound (LeftBound55789.actual selector witness) := by
  exact .transfer (LeftBound55789.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound55789.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound55789.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound55789.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55790

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
