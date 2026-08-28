import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard083
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard084
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard185

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound28081
def owner : Owner := ⟨.program ⟨214⟩, ⟨15596⟩⟩
def transferEvent : Nat := 28081
def frameStart : Nat := 28042
def rule : BoundRule := .identity (.predecessor 0 28080 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28080 .coefficient)
      LeftAuthority28078.bound (LeftAuthority28078.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact28079RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28078.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28078.derived selector witness)

def rawBound : CoeffClass := LeftAuthority28078.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28078.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority28078.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound28081

namespace LeftBound28098
def owner : Owner := ⟨.program ⟨214⟩, ⟨15670⟩⟩
def transferEvent : Nat := 28098
def frameStart : Nat := 28042
def rule : BoundRule := .sum [.predecessor 0 28096 .coefficient, .predecessor 1 28097 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28096 .coefficient)
      LeftBound28081.bound (LeftBound28081.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound28081.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28097 .coefficient)
      LeftAuthority28094.bound (LeftAuthority28094.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority28094.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28081.bound, LeftAuthority28094.bound]
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28081.bound, LeftAuthority28094.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28081.actual selector witness, LeftAuthority28094.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28098

namespace LeftBound28101
def owner : Owner := ⟨.program ⟨214⟩, ⟨15671⟩⟩
def transferEvent : Nat := 28101
def frameStart : Nat := 28042
def rule : BoundRule := .identity (.predecessor 0 28100 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28100 .coefficient)
      LeftBound28098.bound (LeftBound28098.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound28098.derived selector witness)

def rawBound : CoeffClass := LeftBound28098.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28098.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound28098.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound28101

namespace LeftBound28107
def owner : Owner := ⟨.program ⟨214⟩, ⟨15672⟩⟩
def transferEvent : Nat := 28107
def frameStart : Nat := 28042
def rule : BoundRule := .product (.predecessor 0 28105 .coefficient) (.predecessor 1 28106 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28105 .coefficient)
      LeftAuthority28103.bound (LeftAuthority28103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact28104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28103.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28103.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28106 .coefficient)
      LeftBound28101.bound (LeftBound28101.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact28102RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28101.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28101.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority28103.bound LeftBound28101.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28103.bound, LeftBound28101.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority28103.actual selector witness) * (LeftBound28101.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28107

namespace LeftBound28115
def owner : Owner := ⟨.program ⟨214⟩, ⟨15673⟩⟩
def transferEvent : Nat := 28115
def frameStart : Nat := 28042
def rule : BoundRule := .sum [.predecessor 0 28113 .coefficient, .predecessor 1 28114 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28113 .coefficient)
      LeftAuthority28111.bound (LeftAuthority28111.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact28112RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28111.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28111.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28114 .coefficient)
      LeftBound28107.bound (LeftBound28107.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact28109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28107.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28107.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority28111.bound, LeftBound28107.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28111.bound, LeftBound28107.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority28111.actual selector witness, LeftBound28107.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28115

namespace LeftBound28119
def owner : Owner := ⟨.program ⟨214⟩, ⟨27255⟩⟩
def transferEvent : Nat := 28119
def frameStart : Nat := 28042
def rule : BoundRule := .product (.predecessor 0 28117 .coefficient) (.predecessor 1 28118 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28117 .coefficient)
      LeftBound28115.bound (LeftBound28115.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact28116RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28115.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28115.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28118 .coefficient)
      LeftAuthority28092.bound (LeftAuthority28092.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact28093RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28092.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28092.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound28115.bound LeftAuthority28092.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28115.bound, LeftAuthority28092.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound28115.actual selector witness) * (LeftAuthority28092.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28119

namespace LeftBound28130
def owner : Owner := ⟨.program ⟨214⟩, ⟨15639⟩⟩
def transferEvent : Nat := 28130
def frameStart : Nat := 28042
def rule : BoundRule := .product (.predecessor 0 28128 .coefficient) (.predecessor 1 28129 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28128 .coefficient)
      LeftAuthority28103.bound (LeftAuthority28103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact28104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28103.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28103.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28129 .coefficient)
      LeftAuthority28126.bound (LeftAuthority28126.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact28127RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28126.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28126.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority28103.bound LeftAuthority28126.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28103.bound, LeftAuthority28126.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority28103.actual selector witness) * (LeftAuthority28126.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28130

namespace LeftBound28138
def owner : Owner := ⟨.program ⟨214⟩, ⟨15640⟩⟩
def transferEvent : Nat := 28138
def frameStart : Nat := 28042
def rule : BoundRule := .sum [.predecessor 0 28136 .coefficient, .predecessor 1 28137 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28136 .coefficient)
      LeftAuthority28134.bound (LeftAuthority28134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact28135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28134.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28137 .coefficient)
      LeftBound28130.bound (LeftBound28130.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact28132RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28130.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28130.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority28134.bound, LeftBound28130.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28134.bound, LeftBound28130.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority28134.actual selector witness, LeftBound28130.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28138

namespace LeftBound28142
def owner : Owner := ⟨.program ⟨214⟩, ⟨27259⟩⟩
def transferEvent : Nat := 28142
def frameStart : Nat := 28042
def rule : BoundRule := .sum [.predecessor 0 28140 .coefficient, .predecessor 1 28141 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28140 .coefficient)
      LeftBound28138.bound (LeftBound28138.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact28139RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28138.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28138.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28141 .coefficient)
      LeftBound28119.bound (LeftBound28119.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact28124RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28119.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28119.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28138.bound, LeftBound28119.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28138.bound, LeftBound28119.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28138.actual selector witness, LeftBound28119.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28142

namespace LeftBound28155
def owner : Owner := ⟨.program ⟨214⟩, ⟨27257⟩⟩
def transferEvent : Nat := 28155
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 28153 .coefficient, .predecessor 1 28154 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28153 .coefficient)
      LeftBound27984.bound (LeftBound27984.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact28152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27984.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27984.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28154 .coefficient)
      LeftBound27967.bound (LeftBound27967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact27974RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27967.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27967.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27984.bound, LeftBound27967.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27984.bound, LeftBound27967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27984.actual selector witness, LeftBound27967.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28155

namespace LeftBound28158
def owner : Owner := ⟨.program ⟨214⟩, ⟨27257⟩⟩
def transferEvent : Nat := 28158
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 28152 .summary, .result 27974 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28152 .summary)
      LeftBound27986.bound (LeftBound27986.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20983⟩⟩) (rawTerms := some (Proof.Events109.exact28152RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27986.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27974 .summary)
      LeftBound27969.bound (LeftBound27969.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27256⟩⟩) (rawTerms := some (Proof.Events109.exact27974RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27969.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27986.bound, LeftBound27969.bound]
def bound : CoeffClass := .finite ⟨1291978824159503986688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27986.bound, LeftBound27969.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27986.actual selector witness, LeftBound27969.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28158

namespace LeftBound28182
def owner : Owner := ⟨.program ⟨214⟩, ⟨11146⟩⟩
def transferEvent : Nat := 28182
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 28180 .coefficient) (.predecessor 1 28181 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28180 .coefficient)
      LeftAuthority1163.bound (LeftAuthority1163.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1164RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1163.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1163.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28181 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1163.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1163.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1163.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound28182

namespace LeftBound28187
def owner : Owner := ⟨.program ⟨214⟩, ⟨7345⟩⟩
def transferEvent : Nat := 28187
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 28185 .coefficient) (.predecessor 1 28186 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28185 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28186 .coefficient)
      LeftBound13485.bound (LeftBound13485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13485.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13485.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound13485.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound13485.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound13485.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28187

namespace LeftBound28192
def owner : Owner := ⟨.program ⟨214⟩, ⟨11147⟩⟩
def transferEvent : Nat := 28192
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 28190 .coefficient, .predecessor 1 28191 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28190 .coefficient)
      LeftBound28187.bound (LeftBound28187.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28189RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28187.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28187.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28191 .coefficient)
      LeftBound28182.bound (LeftBound28182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28184RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28182.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28182.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28187.bound, LeftBound28182.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28187.bound, LeftBound28182.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28187.actual selector witness, LeftBound28182.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28192

namespace LeftBound28196
def owner : Owner := ⟨.program ⟨214⟩, ⟨11148⟩⟩
def transferEvent : Nat := 28196
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 28194 .coefficient, .predecessor 1 28195 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28194 .coefficient)
      LeftBound28192.bound (LeftBound28192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28195 .coefficient)
      LeftBound13477.bound (LeftBound13477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13477.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28192.bound, LeftBound13477.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28192.bound, LeftBound13477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28192.actual selector witness, LeftBound13477.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28196

namespace LeftBound28197
def owner : Owner := ⟨.program ⟨214⟩, ⟨11148⟩⟩
def transferEvent : Nat := 28197
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨89⟩⟩]⟩ [⟨.result 13478 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13478 .coefficient)
      LeftBound13477.bound (LeftBound13477.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨89⟩⟩) (rawTerms := some (Proof.Events052.exact13478RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13477.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound13477.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound13477.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound28197

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
